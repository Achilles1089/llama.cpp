#pragma once

// the disk tier of the prompt cache: states the RAM prompt cache (server_prompt_cache) cannot hold — too big for it,
// or evicted from it — go to disk as slot files, and a request whose prompt is on disk and not in RAM restores from there
//
// one owner of the facts: this lives inside the server, next to the slots it describes; nothing outside the process
// decides what is cached (ref: docs/CACHE_DESIGN.md Part 2 of the Raven repo this fork serves)
//
// on disk: <dir>/<id>.bin (GGSQ payload + SCKP checkpoint appendix, byte-identical to a /slots save), <id>.bin.draft
// (the SDFT sidecar), <id>.json (the manifest, written LAST: its presence means the entry is valid)

#include "server-common.h"
#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// the slot file encoding, shared by /slots save|restore and the disk tier
namespace server_slot_file {
    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    constexpr uint32_t CKPT_MAGIC   = 0x504b4353; // "SCKP"
    constexpr uint32_t CKPT_VERSION = 1;

    // the draft context (draft model / MTP head KV) is saved next to the slot file, in "<slot file>.draft"
    constexpr uint32_t DRAFT_MAGIC   = 0x54464453; // "SDFT"
    constexpr uint32_t DRAFT_VERSION = 2;           // 2: payload checksum in the header

    std::string draft_path(const std::string & filepath);

    // FNV-1a over the packed token payload of the slot file
    uint64_t tokens_hash(const void * data, size_t size);

    // XXH3 of the sidecar payload: each blob chained through the seed, with its size mixed in
    uint64_t draft_payload_hash(const std::vector<uint8_t> & data_dft, const std::vector<uint8_t> & data_spec);

    // write a GGSQ slot file from a RAM prompt-cache buffer (llama_state_seq_get_data_ext form: an 8-byte
    // [io magic][seq id] header, then the same payload llama_state_seq_save_file writes); returns bytes written, 0 on failure
    size_t write_main_from_buffer(const std::string & path, const std::vector<char> & packed, const std::vector<uint8_t> & main);

    // append the checkpoint appendix; false if it could not be written completely
    bool write_checkpoints(const std::string & path, const std::list<common_prompt_checkpoint> & checkpoints, size_t & n_written);

    // write the draft sidecar (trunc); returns bytes written, 0 on failure (the file is removed then)
    size_t write_draft(const std::string & path, const std::vector<uint8_t> & data_dft, const std::vector<uint8_t> & data_spec,
                       uint64_t tokens_hash, uint64_t n_packed);

    // read the packed tokens from a GGSQ file's header without loading any state; false if the header is not a slot file
    bool read_packed(const std::string & path, llama_tokens & packed);

    // read only the (n_tokens, pos_min, pos_max) of each checkpoint in the appendix at `offset`; false if none usable
    struct ckpt_range { int64_t n_tokens; int32_t pos_min; int32_t pos_max; };
    bool read_checkpoint_ranges(const std::string & path, size_t offset, std::vector<ckpt_range> & out);
}

struct server_prompt_disk_params {
    std::string dir;
    size_t max_bytes      = 0;    // 0 = no budget
    size_t free_min_bytes = 0;    // never fill the volume past this (0 = no floor)
    size_t min_tokens     = 0;    // prompts shorter than this never go to disk
    int    idle_flush_s   = 0;    // 0 = no idle flush
};

struct server_prompt_disk_entry {
    std::string   id;
    server_tokens tokens;                                   // the deserialized packed form (media ids included) — for LCP
    std::vector<server_slot_file::ckpt_range> checkpoints;  // ranges only; the data stays in the file
    size_t  bytes      = 0;                                 // main + draft
    int64_t created_at = 0;
    int64_t used_at    = 0;
    std::string why;
    std::string key;                                        // THE KEY: the conversation (server_prompt::key), "" = unnamed
    int64_t     t_proc = 0;                                 // when its prompt was processed (ms) — the newest of a key wins
};

struct server_prompt_disk_score {
    int   lcp    = 0;   // tokens shared with the request
    int   resume = 0;   // where the hybrid can actually resume: lcp when fully covered, else the newest checkpoint at or below it
    float f_keep = 0.0f;
    float f_sim  = 0.0f;
};

struct server_prompt_disk_hit {
    std::string id;
    std::string path;
    int         n_tokens = 0;
    server_prompt_disk_score score;
};

class server_prompt_cache_disk {
public:
    // `env` is the provenance every manifest must match — the weights, the draft, kv types, context shape, format versions;
    // NEVER the server build (a rebuild must keep the tier). A manifest's extra fields are ignored; a mismatch is another
    // model's state: left alone, never deleted
    server_prompt_cache_disk(server_prompt_disk_params params, json env, bool has_mtmd);
    ~server_prompt_cache_disk();   // drains and joins the writer

    const server_prompt_disk_params params;

    // startup, main thread: read the manifests, verify provenance and sizes, delete what fails, remove *.tmp
    void scan();

    // hooks (a)/(b): a RAM state leaves RAM for disk. Takes ownership; the bytes stay counted (in_flight_bytes) until
    // written. Returns false (and logs why) when refused: below min_tokens, unchanged, or the dir is full with nothing
    // evictable. The writer thread does the I/O.
    bool offer(server_prompt_cache_state && st, std::vector<char> && packed, const char * why);

    // a state the main thread streamed from a slot straight to <id>.bin / .draft (too big for a RAM copy): register it
    void add_streamed(const std::string & id, llama_tokens && packed, size_t payload_end, size_t bytes, const char * why,
                      const std::string & key, int64_t t_proc);

    // bytes handed to the writer and not yet on disk — the RAM tier counts them against its limit
    size_t in_flight_bytes() const { return in_flight.load(); }

    // hook (c): the best entry on disk for `tokens_new`, scored against what the slot already holds — like the RAM
    // tier's find_better, but the hybrid's resume point (a checkpoint at or below the divergence) is what counts
    bool find_better(const server_prompt & prompt, const server_tokens & tokens_new, int n_ctx_slot, server_prompt_disk_hit & hit) const;

    // A STATE IN THE WRITER'S HANDS IS STILL A STATE (2026-10-12 14:0x, live): a chat evicted from RAM is being written
    // for seconds; a request for it in that window used to take an older copy from disk (581 tokens re-read, 9 s).
    // The queued and the in-progress jobs are scored like the disk's entries (hit.path empty) and loaded from memory.
    bool find_better_inflight(const server_prompt & prompt, const server_tokens & tokens_new, int n_ctx_slot, server_prompt_disk_hit & hit) const;
    bool load_inflight(const std::string & id, server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) const;

    bool contains(const std::string & id) const;
    size_t bytes_of(const std::string & id) const;   // 0 when the id is not on the tier
    static constexpr size_t REWRITE_SLACK = 64u << 20;   // a stored copy this much bigger than the new state is rewritten (stage 2)
    void touch(const std::string & id);
    void drop(const std::string & id, const char * why);

    // THE KEY: every state of `key` leaves the tier — the entries (files deleted), the queued writes, and the write in
    // progress once it lands. Returns how many entries and queued jobs went.
    size_t forget_key(const std::string & key);

    // the reserved slot's current prompt: entries that are a prefix of it are never evicted
    void set_protected(server_tokens && tokens);

    // the entry a request is about to restore: held across the outgoing state's save (whose eviction must not take it)
    void set_held(const std::string & id);

    // the main thread saw a request: for the idle flush's clock
    void note_activity();

    // set by the server: called from the writer thread when the server has been idle for idle_flush_s (posts a task)
    std::function<void()> on_idle_flush;
    void flush_done();   // the flush task finished (re-arms the idle flush)

    json to_json() const;

    static std::string id_of(const std::vector<char> & packed);
    static std::string id_of(const llama_tokens & packed);

    std::string path_main (const std::string & id) const;
    std::string path_draft(const std::string & id) const;
    std::string path_json (const std::string & id) const;

private:
    struct job {
        server_prompt_cache_state st;
        std::vector<char> packed;
        std::string id;
        const char * why;
        size_t bytes;
    };

    void writer_loop();
    bool write_one(job & j);
    void register_entry(server_prompt_disk_entry && e, bool write_manifest);
    bool write_manifest(const server_prompt_disk_entry & e) const;
    std::vector<server_prompt_disk_entry> evict_locked(const server_prompt_disk_entry * fresh);   // returns the victims (files to delete)
    bool is_protected_locked(const server_prompt_disk_entry & e) const;
    size_t total_bytes_locked() const;
    size_t free_bytes() const;

    const json env;
    const bool has_mtmd;

    mutable std::mutex m;
    std::vector<server_prompt_disk_entry> entries;
    server_tokens protected_tokens;
    bool has_protected = false;
    std::string held_id;

    std::condition_variable cv;
    std::deque<job> queue;
    const job * writing = nullptr;   // the job under write (the writer's local), readable under `m` while it is set
    std::vector<std::string> forgotten;   // keys forgotten while their state was under write: dropped when it lands
    bool newer_of_key_locked(const std::string & key, int64_t t_proc) const;   // an entry or a queued job of `key` newer than t_proc
    std::thread writer;
    bool stop = false;
    std::atomic<size_t> in_flight{0};
    std::atomic<int64_t> last_activity{0};
    std::atomic<bool> flush_pending{false};
};
