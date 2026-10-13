#include "server-prompt-disk.h"

#include "llama.h"
#include "hash/xxhash/xxhash.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

//
// the slot file encoding
//

namespace server_slot_file {

std::string draft_path(const std::string & filepath) {
    return filepath + ".draft";
}

uint64_t tokens_hash(const void * data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t draft_payload_hash(const std::vector<uint8_t> & data_dft, const std::vector<uint8_t> & data_spec) {
    uint64_t h = XXH3_64bits_withSeed(data_dft.data(),  data_dft.size(),  (uint64_t) data_dft.size());
    h          = XXH3_64bits_withSeed(data_spec.data(), data_spec.size(), h ^ (uint64_t) data_spec.size());
    return h;
}

static void put(std::ofstream & ofs, const void * src, size_t size, size_t & n) {
    ofs.write((const char *) src, size);
    n += size;
}

static void put_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n) {
    const uint64_t len = buf.size();
    put(ofs, &len, sizeof(len), n);
    if (len > 0) {
        put(ofs, buf.data(), len, n);
    }
}

static bool get(std::ifstream & ifs, void * dst, size_t size) {
    return bool(ifs.read((char *) dst, size));
}

size_t write_main_from_buffer(const std::string & path, const std::vector<char> & packed, const std::vector<uint8_t> & main) {
    // the buffer's first 8 bytes are llama's [io magic][seq id]; the file carries neither (ref: llama_state_seq_save_file)
    const size_t io_hdr = sizeof(uint32_t) + sizeof(int32_t);
    if (main.size() < io_hdr || packed.size() % sizeof(llama_token) != 0) {
        return 0;
    }
    std::ofstream ofs(fs::u8path(path), std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return 0;
    }
    size_t n = 0;
    const uint32_t magic   = LLAMA_STATE_SEQ_MAGIC;
    const uint32_t version = LLAMA_STATE_SEQ_VERSION;
    const uint32_t n_tok   = (uint32_t) (packed.size() / sizeof(llama_token));
    put(ofs, &magic,   sizeof(magic),   n);
    put(ofs, &version, sizeof(version), n);
    put(ofs, &n_tok,   sizeof(n_tok),   n);
    put(ofs, packed.data(), packed.size(), n);
    put(ofs, main.data() + io_hdr, main.size() - io_hdr, n);
    ofs.flush();
    return ofs ? n : 0;
}

bool write_checkpoints(const std::string & path, const std::list<common_prompt_checkpoint> & checkpoints, size_t & n_written) {
    n_written = 0;
    if (checkpoints.empty()) {
        return true;
    }
    std::ofstream ofs(fs::u8path(path), std::ios::binary | std::ios::app);
    if (!ofs) {
        return false;
    }
    const uint32_t magic   = CKPT_MAGIC;
    const uint32_t version = CKPT_VERSION;
    const uint32_t count   = (uint32_t) checkpoints.size();
    put(ofs, &magic,   sizeof(magic),   n_written);
    put(ofs, &version, sizeof(version), n_written);
    put(ofs, &count,   sizeof(count),   n_written);
    for (const auto & cur : checkpoints) {
        put(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
        put(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
        put(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
        put_buf(ofs, cur.data_tgt,  n_written);
        put_buf(ofs, cur.data_dft,  n_written);
        put_buf(ofs, cur.data_spec, n_written);
    }
    ofs.flush();
    return bool(ofs);
}

size_t write_draft(const std::string & path, const std::vector<uint8_t> & data_dft, const std::vector<uint8_t> & data_spec,
                   uint64_t tokens_hash, uint64_t n_packed) {
    const uint64_t payload_hash = draft_payload_hash(data_dft, data_spec);
    size_t n = 0;
    {
        std::ofstream ofs(fs::u8path(path), std::ios::binary | std::ios::trunc);
        if (ofs) {
            const uint32_t magic   = DRAFT_MAGIC;
            const uint32_t version = DRAFT_VERSION;
            put(ofs, &magic,        sizeof(magic),        n);
            put(ofs, &version,      sizeof(version),      n);
            put(ofs, &tokens_hash,  sizeof(tokens_hash),  n);
            put(ofs, &n_packed,     sizeof(n_packed),     n);
            put(ofs, &payload_hash, sizeof(payload_hash), n);
            put_buf(ofs, data_dft,  n);
            put_buf(ofs, data_spec, n);
            ofs.flush();
        }
        if (!ofs) {
            n = 0;
        }
    }
    if (n == 0) {
        std::error_code ec;
        fs::remove(fs::u8path(path), ec);
    }
    return n;
}

bool read_packed(const std::string & path, llama_tokens & packed) {
    std::ifstream ifs(fs::u8path(path), std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const size_t file_size = (size_t) ifs.tellg();
    if (!ifs.seekg(0)) {
        return false;
    }
    uint32_t magic = 0, version = 0, n = 0;
    if (!get(ifs, &magic, sizeof(magic)) || !get(ifs, &version, sizeof(version)) || !get(ifs, &n, sizeof(n))) {
        return false;
    }
    if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
        return false;
    }
    if ((size_t) n * sizeof(llama_token) > file_size - 3 * sizeof(uint32_t)) {
        return false;
    }
    packed.resize(n);
    return n == 0 || get(ifs, packed.data(), (size_t) n * sizeof(llama_token));
}

bool read_checkpoint_ranges(const std::string & path, size_t offset, std::vector<ckpt_range> & out) {
    out.clear();
    std::ifstream ifs(fs::u8path(path), std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const size_t file_size = (size_t) ifs.tellg();
    if (file_size < offset || !ifs.seekg(offset)) {
        return false;
    }
    uint32_t magic = 0, version = 0, count = 0;
    if (!get(ifs, &magic, sizeof(magic)) || magic != CKPT_MAGIC ||
        !get(ifs, &version, sizeof(version)) || version != CKPT_VERSION ||
        !get(ifs, &count, sizeof(count))) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        ckpt_range r;
        if (!get(ifs, &r.n_tokens, sizeof(r.n_tokens)) || !get(ifs, &r.pos_min, sizeof(r.pos_min)) || !get(ifs, &r.pos_max, sizeof(r.pos_max))) {
            return false;
        }
        for (int b = 0; b < 3; ++b) {
            uint64_t len = 0;
            if (!get(ifs, &len, sizeof(len)) || len > file_size || !ifs.seekg((std::streamoff) len, std::ios::cur)) {
                return false;
            }
        }
        out.push_back(r);
    }
    return true;
}

} // namespace server_slot_file

//
// the disk tier
//

static int64_t now_s() {
    return (int64_t) std::time(nullptr);
}

std::string server_prompt_cache_disk::id_of(const std::vector<char> & packed) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%016llx-%zu", (unsigned long long) XXH3_64bits(packed.data(), packed.size()), packed.size() / sizeof(llama_token));
    return buf;
}

std::string server_prompt_cache_disk::id_of(const llama_tokens & packed) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%016llx-%zu", (unsigned long long) XXH3_64bits(packed.data(), packed.size() * sizeof(llama_token)), packed.size());
    return buf;
}

std::string server_prompt_cache_disk::path_main (const std::string & id) const { return params.dir + id + ".bin"; }
std::string server_prompt_cache_disk::path_draft(const std::string & id) const { return params.dir + id + ".bin.draft"; }
std::string server_prompt_cache_disk::path_json (const std::string & id) const { return params.dir + id + ".json"; }

server_prompt_cache_disk::server_prompt_cache_disk(server_prompt_disk_params params_, json env_, bool has_mtmd_)
    : params(std::move(params_)), env(std::move(env_)), has_mtmd(has_mtmd_) {
    last_activity = now_s();
    writer = std::thread([this] { writer_loop(); });
}

server_prompt_cache_disk::~server_prompt_cache_disk() {
    {
        std::lock_guard<std::mutex> lock(m);
        stop = true;
    }
    cv.notify_all();
    if (writer.joinable()) {
        writer.join();
    }
}

size_t server_prompt_cache_disk::free_bytes() const {
    std::error_code ec;
    const auto sp = fs::space(fs::u8path(params.dir), ec);
    return ec ? SIZE_MAX : (size_t) sp.available;
}

size_t server_prompt_cache_disk::total_bytes_locked() const {
    size_t n = 0;
    for (const auto & e : entries) {
        n += e.bytes;
    }
    return n;
}

bool server_prompt_cache_disk::is_protected_locked(const server_prompt_disk_entry & e) const {
    // a saved state carries the reply the slot generated after its prompt, and the next request renders that reply
    // through the chat template - so the entry is rarely an exact prefix of the protected prompt: it is the same
    // conversation when all but its generated tail is shared (the tail is bounded by one reply)
    if (!has_protected || e.tokens.size() == 0) {
        return false;
    }
    const size_t lcp   = e.tokens.get_common_prefix(protected_tokens);
    const size_t slack = std::max<size_t>(256, e.tokens.size() / 8);
    return lcp + slack >= e.tokens.size();
}

// SUPERSEDE (2026-10-12 16:10, live: 57 states, 214 GB, ten of them one chat): an older state of the SAME conversation
// is redundant once a longer one exists. "Same conversation" is all but a bounded tail shared — never exact containment:
// a saved state carries what the slot generated after its prompt (a reply as generated, or a touch's user line and
// its one token), and the next state carries that reply as the template renders it, so the older is almost never an
// exact prefix of the newer and the old rule never fired. The tail is bounded by one reply's rendering difference or a
// touch's line (SUPERSEDE_TAIL); it is far smaller than any chat's own content past a shared head, so two chats of one
// project never supersede each other.
static constexpr size_t SUPERSEDE_TAIL = 256;

static bool supersedes(const server_prompt_disk_entry & newer, const server_prompt_disk_entry & older) {
    if (newer.id == older.id) {
        return false;
    }
    // THE KEY (2026-10-12 20:xx, live: 42 states for a dozen chats — the same chat under three heads, a state carrying
    // the generated thinking next to the one carrying it as rendered, 17 of one test chat differing in a clock line):
    // two states of one conversation are versions of it, and the newer processed one is the conversation — whatever
    // their lengths or tails. Named states never fall to the token rule between themselves.
    if (!newer.key.empty() && newer.key == older.key) {
        return newer.t_proc > older.t_proc;
    }
    if (!newer.key.empty() && !older.key.empty()) {
        return false;   // two named conversations are two conversations
    }
    if (!older.key.empty()) {
        return false;   // a named state is never retired by an unnamed one (22:02 scan: a legacy copy, longer, took the live state)
    }
    if (older.tokens.size() == 0) {
        return false;
    }
    if (!newer.key.empty()) {
        // a named state retires an unnamed copy of ITS chat whatever their lengths: the same chat = they share more
        // than the head — the unnamed state's first checkpoint sits at its first user message, and two chats of one
        // surface part within a few tokens after it (the first words, the clock, the folder)
        const size_t head = older.checkpoints.empty() ? 0 : (size_t) std::max<int64_t>(0, older.checkpoints.front().n_tokens);
        const size_t lcp  = older.tokens.get_common_prefix(newer.tokens);
        if (head > 0 && lcp >= head + 64) {
            return true;
        }
    }
    if (newer.tokens.size() <= older.tokens.size()) {
        return false;
    }
    const size_t lcp = older.tokens.get_common_prefix(newer.tokens);
    return lcp + SUPERSEDE_TAIL >= older.tokens.size();
}

std::vector<server_prompt_disk_entry> server_prompt_cache_disk::evict_locked(const server_prompt_disk_entry * fresh) {
    std::vector<server_prompt_disk_entry> victims;
    if (fresh != nullptr) {
        // a fresh write: the older states of its conversation go
        for (auto it = entries.begin(); it != entries.end();) {
            if (supersedes(*fresh, *it)) {
                it->why = "superseded";
                victims.push_back(std::move(*it));
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    } else {
        // the scan at startup: every entry against every longer one — the versions the old rule let pile up
        for (auto it = entries.begin(); it != entries.end();) {
            bool gone = false;
            for (const auto & other : entries) {
                if (supersedes(other, *it)) {
                    gone = true;
                    break;
                }
            }
            if (gone) {
                it->why = "superseded-at-scan";
                victims.push_back(std::move(*it));
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }
    // the budget and the floor: the least recently used first, never the reserved slot's chat
    auto over = [&]() {
        if (params.max_bytes > 0 && total_bytes_locked() > params.max_bytes) {
            return "budget";
        }
        if (params.free_min_bytes > 0 && free_bytes() < params.free_min_bytes) {
            return "free-floor";
        }
        return (const char *) nullptr;
    };
    while (const char * why = over()) {
        auto victim = entries.end();
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (is_protected_locked(*it) || it->id == held_id) {
                continue;
            }
            if (victim == entries.end() || it->used_at < victim->used_at) {
                victim = it;
            }
        }
        if (victim == entries.end()) {
            SRV_WRN("prompt-cache disk: over the %s and nothing evictable (%zu entries, all protected)\n", why, entries.size());
            break;
        }
        victim->why = why;
        victims.push_back(std::move(*victim));
        entries.erase(victim);
    }
    return victims;
}

bool server_prompt_cache_disk::write_manifest(const server_prompt_disk_entry & e) const {
    json ck = json::array();
    for (const auto & c : e.checkpoints) {
        ck.push_back({{"n_tokens", c.n_tokens}, {"pos_min", c.pos_min}, {"pos_max", c.pos_max}});
    }
    std::error_code ec;
    const auto main_bytes  = fs::file_size(fs::u8path(path_main(e.id)), ec);
    const auto draft_bytes = fs::exists(fs::u8path(path_draft(e.id)), ec) ? fs::file_size(fs::u8path(path_draft(e.id)), ec) : 0;
    json j = {
        {"format", 1},
        {"env", env},
        {"id", e.id},
        {"key", e.key},
        {"t_proc", e.t_proc},
        {"n_tokens", (int) e.tokens.size()},
        {"bytes", {{"main", (uint64_t) main_bytes}, {"draft", (uint64_t) draft_bytes}}},
        {"checkpoints", ck},
        {"created_at", e.created_at},
        {"used_at", e.used_at},
        {"why", e.why},
    };
    const std::string tmp = path_json(e.id) + ".tmp";
    {
        std::ofstream ofs(fs::u8path(tmp), std::ios::trunc);
        if (!ofs) {
            return false;
        }
        ofs << j.dump();
        ofs.flush();
        if (!ofs) {
            return false;
        }
    }
    fs::rename(fs::u8path(tmp), fs::u8path(path_json(e.id)), ec);
    return !ec;
}

void server_prompt_cache_disk::register_entry(server_prompt_disk_entry && e, bool manifest) {
    if (manifest && !write_manifest(e)) {
        SRV_ERR("prompt-cache disk: could not write the manifest of %s — the entry is dropped\n", e.id.c_str());
        std::error_code ec;
        fs::remove(fs::u8path(path_main(e.id)), ec);
        fs::remove(fs::u8path(path_draft(e.id)), ec);
        return;
    }
    std::vector<server_prompt_disk_entry> victims;
    {
        std::lock_guard<std::mutex> lock(m);
        // THE KEY: forgotten while under write — the files go, the row is never made
        if (!e.key.empty()) {
            auto f = std::find(forgotten.begin(), forgotten.end(), e.key);
            if (f != forgotten.end()) {
                forgotten.erase(f);
                std::error_code ec;
                fs::remove(fs::u8path(path_json(e.id)), ec);
                fs::remove(fs::u8path(path_main(e.id)), ec);
                fs::remove(fs::u8path(path_draft(e.id)), ec);
                SRV_INF("prompt-cache disk: dropped id=%s key=%s why=forgotten-under-write\n", e.id.c_str(), e.key.c_str());
                return;
            }
        }
        // the same id again (a rewrite): replace the row
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->id == e.id) {
                entries.erase(it);
                break;
            }
        }
        entries.push_back(std::move(e));
        victims = evict_locked(&entries.back());
    }
    for (const auto & v : victims) {
        std::error_code ec;
        fs::remove(fs::u8path(path_json(v.id)), ec);   // the manifest first: a half-deleted entry is never offered
        fs::remove(fs::u8path(path_main(v.id)), ec);
        fs::remove(fs::u8path(path_draft(v.id)), ec);
        SRV_INF("prompt-cache disk: evicted id=%s tokens=%zu bytes=%zu why=%s\n", v.id.c_str(), v.tokens.size(), v.bytes, v.why.c_str());
    }
}

bool server_prompt_cache_disk::offer(server_prompt_cache_state && st, std::vector<char> && packed, const char * why) {
    const size_t n_tokens = st.prompt.n_tokens();
    if (n_tokens < params.min_tokens) {
        SRV_INF("prompt-cache disk: refused tokens=%zu why=min-tokens (%zu) [%s]\n", n_tokens, params.min_tokens, why);
        return false;
    }
    const std::string id = id_of(packed);
    const size_t bytes = st.size();
    const size_t have = bytes_of(id);
    if (have > 0) {
        if (have <= bytes + REWRITE_SLACK) {
            touch(id);
            SRV_INF("prompt-cache disk: unchanged id=%s tokens=%zu, not rewritten [%s]\n", id.c_str(), n_tokens, why);
            return false;
        }
        drop(id, "rewritten smaller (stage 2: a leaner checkpoint appendix)");   // the same tokens, a leaner state
    }
    {
        std::lock_guard<std::mutex> lock(m);
        if (stop) {
            return false;
        }
        // THE KEY: a state evicted from RAM after a newer state of its chat already reached the tier (or its queue) is
        // stale — not written. And the queued older states of this key are dropped before they cost a write.
        if (!st.prompt.key.empty()) {
            if (newer_of_key_locked(st.prompt.key, st.prompt.t_proc)) {
                SRV_INF("prompt-cache disk: refused id=%s key=%s tokens=%zu why=stale (a newer state of the key is on the tier) [%s]\n",
                        id.c_str(), st.prompt.key.c_str(), n_tokens, why);
                return false;
            }
            for (auto it = queue.begin(); it != queue.end();) {
                if (it->st.prompt.key == st.prompt.key && it->st.prompt.t_proc < st.prompt.t_proc) {
                    SRV_INF("prompt-cache disk: a queued older state of key %s (%d tokens) is superseded before its write\n",
                            st.prompt.key.c_str(), it->st.prompt.n_tokens());
                    in_flight -= it->bytes;
                    it = queue.erase(it);
                } else {
                    ++it;
                }
            }
        }
        in_flight += bytes;
        queue.push_back(job{std::move(st), std::move(packed), id, why, bytes});
    }
    cv.notify_one();
    return true;
}

bool server_prompt_cache_disk::write_one(job & j) {
    const int64_t t0 = ggml_time_us();
    const std::string tmp_main  = path_main(j.id) + ".tmp";
    const std::string tmp_draft = path_draft(j.id) + ".tmp";
    std::error_code ec;

    const size_t n_main = server_slot_file::write_main_from_buffer(tmp_main, j.packed, j.st.data.main);
    if (n_main == 0) {
        SRV_ERR("prompt-cache disk: could not write %s\n", tmp_main.c_str());
        fs::remove(fs::u8path(tmp_main), ec);
        return false;
    }
    size_t n_ckpt = 0;
    if (!server_slot_file::write_checkpoints(tmp_main, j.st.prompt.checkpoints, n_ckpt)) {
        SRV_ERR("prompt-cache disk: could not append the checkpoints to %s\n", tmp_main.c_str());
        fs::remove(fs::u8path(tmp_main), ec);
        return false;
    }
    size_t n_draft = 0;
    if (!j.st.data.drft.empty()) {
        const std::vector<uint8_t> no_spec;   // the RAM tier carries no speculative state; the loader resets it
        n_draft = server_slot_file::write_draft(tmp_draft, j.st.data.drft, no_spec,
                                                server_slot_file::tokens_hash(j.packed.data(), j.packed.size()),
                                                j.packed.size() / sizeof(llama_token));
    }
    fs::rename(fs::u8path(tmp_main), fs::u8path(path_main(j.id)), ec);
    if (ec) {
        SRV_ERR("prompt-cache disk: could not place %s\n", path_main(j.id).c_str());
        fs::remove(fs::u8path(tmp_main), ec);
        fs::remove(fs::u8path(tmp_draft), ec);
        return false;
    }
    if (n_draft > 0) {
        fs::rename(fs::u8path(tmp_draft), fs::u8path(path_draft(j.id)), ec);
    } else {
        fs::remove(fs::u8path(path_draft(j.id)), ec);
    }

    server_prompt_disk_entry e;
    e.id = j.id;
    {
        llama_tokens ids(j.packed.size() / sizeof(llama_token));
        memcpy(ids.data(), j.packed.data(), j.packed.size());
        e.tokens = server_tokens::deserialize(ids, has_mtmd);
    }
    for (const auto & c : j.st.prompt.checkpoints) {
        e.checkpoints.push_back({c.n_tokens, c.pos_min, c.pos_max});
    }
    e.bytes      = n_main + n_ckpt + n_draft;
    e.created_at = e.used_at = now_s();
    e.why        = j.why;
    e.key        = j.st.prompt.key;
    e.t_proc     = j.st.prompt.t_proc;
    const size_t n_tokens = e.tokens.size();
    const size_t bytes    = e.bytes;
    const size_t n_ck     = e.checkpoints.size();
    register_entry(std::move(e), true);
    SRV_INF("prompt-cache disk: saved id=%s key=%s tokens=%zu bytes=%zu ckpts=%zu draft=%zu t=%.0fms why=%s\n",
            j.id.c_str(), j.st.prompt.key.c_str(), n_tokens, bytes, n_ck, n_draft, (ggml_time_us() - t0) / 1000.0, j.why);
    return true;
}

void server_prompt_cache_disk::writer_loop() {
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lock(m);
            if (!cv.wait_for(lock, std::chrono::seconds(60), [&] { return stop || !queue.empty(); })) {
                // a quiet minute: is the server idle long enough for a flush?
                const bool idle = params.idle_flush_s > 0 && on_idle_flush && queue.empty() && !flush_pending.load()
                               && now_s() - last_activity.load() >= params.idle_flush_s;
                lock.unlock();
                if (idle) {
                    flush_pending = true;
                    on_idle_flush();
                }
                continue;
            }
            if (stop && queue.empty()) {
                return;
            }
            j = std::move(queue.front());
            queue.pop_front();
            writing = &j;             // a request may still load this state from memory while it is written
        }
        try {
            write_one(j);
        } catch (const std::exception & e) {
            SRV_ERR("prompt-cache disk: write of %s failed: %s\n", j.id.c_str(), e.what());
        }
        {
            std::lock_guard<std::mutex> lock(m);   // a load in progress holds `m`: it finishes before `j` dies
            writing = nullptr;
        }
        in_flight -= j.bytes;
    }
}

static bool score_state(const server_prompt & prompt, const server_tokens & tokens_new, int n_ctx_slot,
                        const server_prompt_cache_state & st, int lcp_slot, float & f_keep_best, float & f_sim_best,
                        server_prompt_disk_score & score) {
    const auto & toks = st.prompt.tokens;
    if (toks.size() == 0 || (int) toks.size() > n_ctx_slot) {
        return false;
    }
    const int lcp = (int) toks.get_common_prefix(tokens_new);
    int resume = 0;
    if (lcp == (int) toks.size()) {
        resume = lcp;
    } else {
        for (const auto & c : st.prompt.checkpoints) {
            if ((int) c.n_tokens <= lcp && (int) c.n_tokens > resume) {
                resume = (int) c.n_tokens;
            }
        }
    }
    const float f_keep = float(lcp) / toks.size();
    const float f_sim  = float(resume) / tokens_new.size();
    if (f_keep < 0.25f) {
        return false;
    }
    if (resume <= lcp_slot + std::max(64, (int) tokens_new.size() / 100)) {
        return false;
    }
    if (f_keep_best < f_keep && f_sim_best < f_sim) {
        f_keep_best = f_keep;
        f_sim_best  = f_sim;
        score       = {lcp, resume, f_keep, f_sim};
        return true;
    }
    (void) prompt;
    return false;
}

bool server_prompt_cache_disk::find_better_inflight(const server_prompt & prompt, const server_tokens & tokens_new, int n_ctx_slot, server_prompt_disk_hit & hit) const {
    if (tokens_new.size() == 0) {
        return false;
    }
    const int lcp_slot = (int) prompt.tokens.get_common_prefix(tokens_new);
    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_slot) / prompt.tokens.size() : -1.0f;
    float f_sim_best  = float(lcp_slot) / tokens_new.size();

    std::lock_guard<std::mutex> lock(m);
    const job * best = nullptr;
    server_prompt_disk_score best_score;
    for (const auto & j : queue) {
        if (score_state(prompt, tokens_new, n_ctx_slot, j.st, lcp_slot, f_keep_best, f_sim_best, best_score)) {
            best = &j;
        }
    }
    if (writing != nullptr && score_state(prompt, tokens_new, n_ctx_slot, writing->st, lcp_slot, f_keep_best, f_sim_best, best_score)) {
        best = writing;
    }
    if (best == nullptr) {
        return false;
    }
    hit.id       = best->id;
    hit.path     = "";
    hit.n_tokens = (int) best->st.prompt.tokens.size();
    hit.score    = best_score;
    return true;
}

bool server_prompt_cache_disk::load_inflight(const std::string & id, server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) const {
    std::lock_guard<std::mutex> lock(m);   // held across the copy: the writer never frees the job under us
    const job * j = nullptr;
    for (const auto & q : queue) {
        if (q.id == id) {
            j = &q;
            break;
        }
    }
    if (j == nullptr && writing != nullptr && writing->id == id) {
        j = writing;
    }
    if (j == nullptr) {
        return false;                      // written and gone from the writer: the disk tier has it now
    }
    {
        const auto & data = j->st.data.main;
        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), data.size(), id_slot, 0);
        if (n != data.size()) {
            SRV_ERR("prompt-cache disk: an in-flight state could not be loaded (%zu of %zu bytes)\n", n, data.size());
            return false;
        }
    }
    if (!j->st.data.drft.empty() && ctx_dft != nullptr) {
        const auto & data = j->st.data.drft;
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), data.size(), id_slot, 0);
        if (n != data.size()) {
            SRV_WRN("prompt-cache disk: an in-flight draft state could not be loaded (%zu of %zu bytes)\n", n, data.size());
            return false;
        }
    }
    prompt = j->st.prompt.clone();         // the writer keeps its copy for the file
    return true;
}

void server_prompt_cache_disk::add_streamed(const std::string & id, llama_tokens && packed, size_t payload_end, size_t bytes, const char * why,
                                           const std::string & key, int64_t t_proc) {
    server_prompt_disk_entry e;
    e.id = id;
    e.key = key;
    e.t_proc = t_proc;
    try {
        e.tokens = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & ex) {
        SRV_ERR("prompt-cache disk: %s: %s — not indexed\n", id.c_str(), ex.what());
        return;
    }
    server_slot_file::read_checkpoint_ranges(path_main(id), payload_end, e.checkpoints);
    e.bytes      = bytes;
    e.created_at = e.used_at = now_s();
    e.why        = why;
    const size_t n_tokens = e.tokens.size();
    const size_t n_ck     = e.checkpoints.size();
    register_entry(std::move(e), true);
    SRV_INF("prompt-cache disk: saved id=%s key=%s tokens=%zu bytes=%zu ckpts=%zu (streamed) why=%s\n", id.c_str(), key.c_str(), n_tokens, bytes, n_ck, why);
}

void server_prompt_cache_disk::scan() {
    std::error_code ec;
    fs::create_directories(fs::u8path(params.dir), ec);
    size_t n_ok = 0, n_dropped = 0, n_tmp = 0, n_foreign = 0;
    std::vector<server_prompt_disk_entry> found;
    for (const auto & de : fs::directory_iterator(fs::u8path(params.dir), ec)) {
        const std::string name = de.path().filename().u8string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            fs::remove(de.path(), ec);
            n_tmp++;
            continue;
        }
        if (name.size() <= 5 || name.compare(name.size() - 5, 5, ".json") != 0) {
            continue;
        }
        const std::string id = name.substr(0, name.size() - 5);
        auto drop = [&](const char * why) {
            SRV_WRN("prompt-cache disk: dropped id=%s why=%s\n", id.c_str(), why);
            fs::remove(fs::u8path(path_json(id)), ec);
            fs::remove(fs::u8path(path_main(id)), ec);
            fs::remove(fs::u8path(path_draft(id)), ec);
            n_dropped++;
        };
        json j;
        try {
            std::ifstream ifs(de.path());
            std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            j = json::parse(text);
        } catch (...) {
            drop("corrupt manifest");
            continue;
        }
        if (j.value("format", 0) != 1) {
            drop("format");
            continue;
        }
        {
            // provenance: every field THIS lead requires must match; fields a manifest carries beyond them (an older
            // lead's "build") are ignored. A mismatch is another model's state (or an older format's): not ours to
            // load — and not ours to delete (2026-10-11: a rebuild's build id dropped every file; a second model on
            // the same box would wipe the first's). It stays, uncounted, until its own lead reads or evicts it.
            const json menv = j.value("env", json::object());
            bool ours = menv.is_object();
            for (auto it = env.begin(); ours && it != env.end(); ++it) {
                ours = menv.contains(it.key()) && menv[it.key()] == it.value();
            }
            if (!ours) {
                n_foreign++;
                continue;
            }
        }
        const uint64_t want_main  = j.value("bytes", json::object()).value("main", (uint64_t) 0);
        const uint64_t want_draft = j.value("bytes", json::object()).value("draft", (uint64_t) 0);
        const auto main_bytes  = fs::exists(fs::u8path(path_main(id)), ec) ? fs::file_size(fs::u8path(path_main(id)), ec) : 0;
        const auto draft_bytes = fs::exists(fs::u8path(path_draft(id)), ec) ? fs::file_size(fs::u8path(path_draft(id)), ec) : 0;
        if (main_bytes == 0 || main_bytes != want_main || draft_bytes != want_draft) {
            drop("size");
            continue;
        }
        llama_tokens packed;
        if (!server_slot_file::read_packed(path_main(id), packed) || id_of(packed) != id) {
            drop("header");
            continue;
        }
        server_prompt_disk_entry e;
        e.id = id;
        try {
            e.tokens = server_tokens::deserialize(packed, has_mtmd);
        } catch (const std::exception &) {
            drop("media without mmproj");
            continue;
        }
        for (const auto & c : j.value("checkpoints", json::array())) {
            e.checkpoints.push_back({c.value("n_tokens", (int64_t) 0), c.value("pos_min", 0), c.value("pos_max", 0)});
        }
        e.bytes      = main_bytes + draft_bytes;
        e.created_at = j.value("created_at", now_s());
        e.used_at    = j.value("used_at", now_s());
        e.why        = j.value("why", "");
        e.key        = j.value("key", "");
        e.t_proc     = j.value("t_proc", (int64_t) 0);
        found.push_back(std::move(e));
        n_ok++;
    }
    std::vector<server_prompt_disk_entry> victims;
    {
        std::lock_guard<std::mutex> lock(m);
        entries = std::move(found);
        victims = evict_locked(nullptr);
    }
    for (const auto & v : victims) {
        fs::remove(fs::u8path(path_json(v.id)), ec);
        fs::remove(fs::u8path(path_main(v.id)), ec);
        fs::remove(fs::u8path(path_draft(v.id)), ec);
        SRV_INF("prompt-cache disk: evicted id=%s tokens=%zu bytes=%zu why=%s\n", v.id.c_str(), v.tokens.size(), v.bytes, v.why.c_str());
    }
    std::lock_guard<std::mutex> lock(m);
    SRV_INF("prompt-cache disk: %s — %zu entr%s (%.1f GiB), %zu dropped, %zu foreign left alone, %zu tmp removed, budget %.0f GiB, floor %.0f GiB free\n",
            params.dir.c_str(), entries.size(), entries.size() == 1 ? "y" : "ies", total_bytes_locked() / (1024.0 * 1024.0 * 1024.0),
            n_dropped, n_foreign, n_tmp, params.max_bytes / (1024.0 * 1024.0 * 1024.0), params.free_min_bytes / (1024.0 * 1024.0 * 1024.0));
}

bool server_prompt_cache_disk::find_better(const server_prompt & prompt, const server_tokens & tokens_new, int n_ctx_slot, server_prompt_disk_hit & hit) const {
    if (tokens_new.size() == 0) {
        return false;
    }
    const int lcp_slot = (int) prompt.tokens.get_common_prefix(tokens_new);
    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_slot) / prompt.tokens.size() : -1.0f;   // empty slot: any entry wins
    float f_sim_best  = float(lcp_slot) / tokens_new.size();

    std::lock_guard<std::mutex> lock(m);
    const server_prompt_disk_entry * best = nullptr;
    server_prompt_disk_score best_score;
    for (const auto & e : entries) {
        if (e.tokens.size() == 0 || (int) e.tokens.size() > n_ctx_slot) {
            continue;
        }
        const int lcp = (int) e.tokens.get_common_prefix(tokens_new);
        // the hybrid's real resume point: the whole entry when fully covered, else the newest checkpoint at or below the divergence
        int resume = 0;
        if (lcp == (int) e.tokens.size()) {
            resume = lcp;
        } else {
            for (const auto & c : e.checkpoints) {
                if (c.n_tokens <= lcp && (int) c.n_tokens > resume) {
                    resume = (int) c.n_tokens;
                }
            }
        }
        const float f_keep = float(lcp) / e.tokens.size();
        const float f_sim  = float(resume) / tokens_new.size();
        if (f_keep < 0.25f) {
            continue;   // don't trash large prompts
        }
        // a restore reads GBs from the SSD: it must gain more than a sliver over what the slot already holds
        // (2026-10-11: slot 27305 vs entry 27306 — 3 GB read to save one token)
        if (resume <= lcp_slot + std::max(64, (int) tokens_new.size() / 100)) {
            continue;
        }
        if (f_keep_best < f_keep && f_sim_best < f_sim) {
            f_keep_best = f_keep;
            f_sim_best  = f_sim;
            best        = &e;
            best_score  = {lcp, resume, f_keep, f_sim};
        }
    }
    if (best == nullptr) {
        return false;
    }
    hit.id       = best->id;
    hit.path     = path_main(best->id);
    hit.n_tokens = (int) best->tokens.size();
    hit.score    = best_score;
    return true;
}

size_t server_prompt_cache_disk::bytes_of(const std::string & id) const {
    std::lock_guard<std::mutex> lock(m);
    for (const auto & e : entries) {
        if (e.id == id) {
            return e.bytes;
        }
    }
    return 0;
}

bool server_prompt_cache_disk::contains(const std::string & id) const {
    std::lock_guard<std::mutex> lock(m);
    for (const auto & e : entries) {
        if (e.id == id) {
            return true;
        }
    }
    return false;
}

void server_prompt_cache_disk::touch(const std::string & id) {
    std::lock_guard<std::mutex> lock(m);
    for (auto & e : entries) {
        if (e.id == id) {
            e.used_at = now_s();
            return;
        }
    }
}

void server_prompt_cache_disk::drop(const std::string & id, const char * why) {
    {
        std::lock_guard<std::mutex> lock(m);
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->id == id) {
                entries.erase(it);
                break;
            }
        }
    }
    std::error_code ec;
    fs::remove(fs::u8path(path_json(id)), ec);
    fs::remove(fs::u8path(path_main(id)), ec);
    fs::remove(fs::u8path(path_draft(id)), ec);
    SRV_WRN("prompt-cache disk: dropped id=%s why=%s\n", id.c_str(), why);
}

bool server_prompt_cache_disk::newer_of_key_locked(const std::string & key, int64_t t_proc) const {
    for (const auto & e : entries) {
        if (e.key == key && e.t_proc > t_proc) {
            return true;
        }
    }
    for (const auto & j : queue) {
        if (j.st.prompt.key == key && j.st.prompt.t_proc > t_proc) {
            return true;
        }
    }
    if (writing != nullptr && writing->st.prompt.key == key && writing->st.prompt.t_proc > t_proc) {
        return true;
    }
    return false;
}

size_t server_prompt_cache_disk::forget_key(const std::string & key) {
    if (key.empty()) {
        return 0;
    }
    std::vector<server_prompt_disk_entry> victims;
    size_t n_queued = 0;
    {
        std::lock_guard<std::mutex> lock(m);
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->key == key) {
                it->why = "forgotten";
                victims.push_back(std::move(*it));
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = queue.begin(); it != queue.end();) {
            if (it->st.prompt.key == key) {
                in_flight -= it->bytes;
                it = queue.erase(it);
                n_queued++;
            } else {
                ++it;
            }
        }
        if (writing != nullptr && writing->st.prompt.key == key) {
            forgotten.push_back(key);   // it lands, then it goes (register_entry)
        }
    }
    for (const auto & v : victims) {
        std::error_code ec;
        fs::remove(fs::u8path(path_json(v.id)), ec);
        fs::remove(fs::u8path(path_main(v.id)), ec);
        fs::remove(fs::u8path(path_draft(v.id)), ec);
        SRV_INF("prompt-cache disk: evicted id=%s key=%s tokens=%zu bytes=%zu why=forgotten\n", v.id.c_str(), key.c_str(), v.tokens.size(), v.bytes);
    }
    return victims.size() + n_queued;
}

void server_prompt_cache_disk::set_held(const std::string & id) {
    std::lock_guard<std::mutex> lock(m);
    held_id = id;
}

void server_prompt_cache_disk::set_protected(server_tokens && tokens) {
    std::lock_guard<std::mutex> lock(m);
    protected_tokens = std::move(tokens);
    has_protected = protected_tokens.size() > 0;
}

void server_prompt_cache_disk::note_activity() {
    last_activity = now_s();
}

void server_prompt_cache_disk::flush_done() {
    flush_pending = false;
    last_activity = now_s();
}

json server_prompt_cache_disk::to_json() const {
    std::lock_guard<std::mutex> lock(m);
    json rows = json::array();
    for (const auto & e : entries) {
        rows.push_back({
            {"id", e.id}, {"key", e.key}, {"t_proc", e.t_proc}, {"n_tokens", (int) e.tokens.size()}, {"bytes", e.bytes},
            {"checkpoints", e.checkpoints.size()},
            {"created_at", e.created_at}, {"used_at", e.used_at}, {"why", e.why}, {"protected", is_protected_locked(e)},
        });
    }
    return {
        {"dir", params.dir}, {"entries", rows}, {"total_bytes", total_bytes_locked()}, {"in_flight_bytes", in_flight.load()},
        {"max_bytes", params.max_bytes}, {"free_min_bytes", params.free_min_bytes}, {"free_bytes", free_bytes()},
        {"min_tokens", params.min_tokens}, {"idle_flush_s", params.idle_flush_s},
    };
}
