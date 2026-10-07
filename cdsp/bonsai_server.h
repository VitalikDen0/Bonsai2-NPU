#ifndef BONSAI_SERVER_H
#define BONSAI_SERVER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#endif

// Special token IDs
#define TOK_ENDOFTEXT 248044
#define TOK_IM_START  248045
#define TOK_IM_END    248046

static void json_escape(const char* in, int in_len, char* out, size_t out_cap) {
    size_t o = 0;
    for (int i = 0; i < in_len && o + 8 < out_cap; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"') { out[o++] = '\\'; out[o++] = '"'; }
        else if (c == '\\') { out[o++] = '\\'; out[o++] = '\\'; }
        else if (c == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (c == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (c == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (c < 32) {
            o += snprintf(out + o, out_cap - o, "\\u%04x", c);
        } else {
            out[o++] = c;
        }
    }
    out[o] = '\0';
}

// Simple JSON extraction helpers
static const char* find_json_field(const char* json, const char* field) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", field);
    const char* p = strstr(json, needle);
    if (!p) return NULL;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':')) p++;
    return p;
}

static int parse_json_bool(const char* json, const char* field, int def_val) {
    const char* p = find_json_field(json, field);
    if (!p) return def_val;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def_val;
}

static int parse_json_int(const char* json, const char* field, int def_val) {
    const char* p = find_json_field(json, field);
    if (!p) return def_val;
    return atoi(p);
}

typedef struct {
    char role[16];
    char* content;
} ServerChatMessage;

static int parse_chat_messages(const char* json, ServerChatMessage* msgs, int max_msgs) {
    const char* p = strstr(json, "\"messages\"");
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    p++; // past '['

    int count = 0;
    while (*p && *p != ']' && count < max_msgs) {
        const char* obj_start = strchr(p, '{');
        if (!obj_start) break;
        const char* obj_end = strchr(obj_start, '}');
        if (!obj_end) break;

        // Parse role
        msgs[count].role[0] = '\0';
        const char* r = strstr(obj_start, "\"role\"");
        if (r && r < obj_end) {
            const char* q1 = strchr(r + 6, '"');
            if (q1 && q1 < obj_end) {
                const char* q2 = strchr(q1 + 1, '"');
                if (q2 && q2 < obj_end) {
                    int rlen = (int)(q2 - q1 - 1);
                    if (rlen > 15) rlen = 15;
                    strncpy(msgs[count].role, q1 + 1, rlen);
                    msgs[count].role[rlen] = '\0';
                }
            }
        }

        // Parse content
        msgs[count].content = NULL;
        const char* c = strstr(obj_start, "\"content\"");
        if (c && c < obj_end) {
            const char* q1 = strchr(c + 9, '"');
            if (q1 && q1 < obj_end) {
                // Find matching end quote taking escapes into account
                const char* cur = q1 + 1;
                while (cur < obj_end) {
                    if (*cur == '"' && *(cur - 1) != '\\') break;
                    cur++;
                }
                if (cur <= obj_end) {
                    int clen = (int)(cur - q1 - 1);
                    char* buf = (char*)malloc(clen + 1);
                    if (buf) {
                        int out_idx = 0;
                        for (int i = 0; i < clen; i++) {
                            if (q1[1 + i] == '\\' && i + 1 < clen) {
                                char next = q1[1 + i + 1];
                                if (next == 'n') { buf[out_idx++] = '\n'; i++; }
                                else if (next == 'r') { buf[out_idx++] = '\r'; i++; }
                                else if (next == 't') { buf[out_idx++] = '\t'; i++; }
                                else if (next == '"') { buf[out_idx++] = '"'; i++; }
                                else if (next == '\\') { buf[out_idx++] = '\\'; i++; }
                                else { buf[out_idx++] = next; i++; }
                            } else {
                                buf[out_idx++] = q1[1 + i];
                            }
                        }
                        buf[out_idx] = '\0';
                        msgs[count].content = buf;
                    }
                }
            }
        }

        if (msgs[count].role[0] && msgs[count].content) {
            count++;
        }
        p = obj_end + 1;
    }
    return count;
}

static void free_chat_messages(ServerChatMessage* msgs, int count) {
    for (int i = 0; i < count; i++) {
        if (msgs[i].content) {
            free(msgs[i].content);
            msgs[i].content = NULL;
        }
    }
}

// Format Bonsai 2 ChatML template:
// <|im_start|>system\n...<|im_end|>\n
// <|im_start|>user\n...<|im_end|>\n
// <|im_start|>assistant\n
static int format_bonsai_prompt(ServerChatMessage* msgs, int count, char* out_prompt, size_t out_cap) {
    size_t o = 0;
    int has_system = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(msgs[i].role, "system") == 0) {
            has_system = 1;
            break;
        }
    }

    if (!has_system) {
        o += snprintf(out_prompt + o, out_cap - o, "<|im_start|>system\nYou are a helpful, respectful, and honest AI assistant.<|im_end|>\n");
    }

    for (int i = 0; i < count; i++) {
        o += snprintf(out_prompt + o, out_cap - o, "<|im_start|>%s\n%s<|im_end|>\n", msgs[i].role, msgs[i].content);
        if (o >= out_cap - 64) break;
    }

    o += snprintf(out_prompt + o, out_cap - o, "<|im_start|>assistant\n");
    return (int)o;
}

// High-speed token generation engine callback
typedef void (*TokenStreamCallback)(int tok_id, const char* tok_str, int is_final, void* user_arg);

static int generate_stream(const char* prompt, int max_gen_tokens, float* logits_buf,
                           TokenStreamCallback cb, void* user_arg) {
    reset_engine_states();

    static int prompt_ids[65536];
    int nids = tok_encode((const unsigned char*)prompt, (int)strlen(prompt), prompt_ids, 65536);
    if (nids <= 0) return -1;

    int n_prefill = nids - 1;
    int p_pos = 0;
    while (p_pos < n_prefill) {
        int chunk = n_prefill - p_pos;
        if (chunk > 1) chunk = 1;
        if (forward_tokens_batch(prompt_ids + p_pos, p_pos, chunk, g_hidden_batch)) {
            return -2;
        }
        p_pos += chunk;
    }

    int cur = prompt_ids[nids - 1];
    int gen_count = 0;

    for (int s = 0; s < max_gen_tokens; s++) {
        int pos = (nids - 1) + s;
        if (pos >= g_ctx - 1) break;

        if (forward_tokens_batch(&cur, pos, 1, hidden)) break;
        if (cdsp_lmhead(hidden, logits_buf)) break;

        int bi = 0;
        if (g_fast_argmax_idx >= 0 && g_fast_argmax_idx < 248320) {
            bi = g_fast_argmax_idx;
        } else {
            for (int i = 0; i < 248320; i++) {
                if (logits_buf[i] > logits_buf[bi]) bi = i;
            }
        }

        if (bi == TOK_IM_END || bi == TOK_ENDOFTEXT) {
            if (cb) cb(bi, "", 1, user_arg);
            break;
        }

        unsigned char dec[512];
        int nb = tok_decode(&bi, 1, dec, sizeof(dec) - 1);
        if (nb < 0) nb = 0;
        dec[nb] = '\0';

        gen_count++;
        if (cb) cb(bi, (const char*)dec, 0, user_arg);
        cur = bi;
    }

    return gen_count;
}

// -------------------------------------------------------------
// Interactive Terminal REPL Mode (--chat)
// -------------------------------------------------------------
static void repl_token_cb(int tok_id, const char* tok_str, int is_final, void* user_arg) {
    (void)tok_id; (void)user_arg;
    if (!is_final) {
        printf("%s", tok_str);
        fflush(stdout);
    }
}

static void run_interactive_chat(float* logits_buf) {
    printf("\n");
    printf("====================================================================\n");
    printf("  Bonsai 2 27B NPU Orchestrator - Interactive Chat REPL\n");
    printf("  Target: Snapdragon 8 Elite (CDSP v79 HVX, Ternary Q2 + Hadamard)\n");
    printf("  Commands: type your message, 'clear' to reset, or 'exit' to quit.\n");
    printf("====================================================================\n\n");

    char conv_prompt[65536];
    conv_prompt[0] = '\0';
    size_t conv_len = 0;
    conv_len += snprintf(conv_prompt + conv_len, sizeof(conv_prompt) - conv_len,
                        "<|im_start|>system\nYou are a helpful, brilliant AI assistant powered by Qualcomm Hexagon NPU.<|im_end|>\n");

    char line[4096];
    while (1) {
        printf("\033[1;32mUser>\033[0m ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        // Strip trailing newline
        size_t slen = strlen(line);
        while (slen > 0 && (line[slen - 1] == '\n' || line[slen - 1] == '\r')) line[--slen] = '\0';
        if (slen == 0) continue;

        if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) {
            printf("[fwd] Exiting chat.\n");
            break;
        }
        if (strcmp(line, "clear") == 0) {
            conv_len = 0;
            conv_len += snprintf(conv_prompt + conv_len, sizeof(conv_prompt) - conv_len,
                                "<|im_start|>system\nYou are a helpful, brilliant AI assistant powered by Qualcomm Hexagon NPU.<|im_end|>\n");
            printf("[fwd] Conversation history cleared.\n\n");
            continue;
        }

        // Append user turn
        conv_len += snprintf(conv_prompt + conv_len, sizeof(conv_prompt) - conv_len,
                            "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);

        printf("\033[1;36mBonsai 2>\033[0m ");
        fflush(stdout);

        uint64_t t0 = 0;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        int gen_tokens = generate_stream(conv_prompt, 1024, logits_buf, repl_token_cb, NULL);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double elapsed_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1000000.0;
        double tok_s = gen_tokens > 0 ? (gen_tokens * 1000.0 / elapsed_ms) : 0.0;

        printf("\n\033[2m[%d tokens generated in %.1f ms | %.2f tok/s]\033[0m\n\n", gen_tokens, elapsed_ms, tok_s);

        // Cap conversation history to prevent buffer overflow
        if (conv_len > 48000) {
            conv_len = 0;
            conv_len += snprintf(conv_prompt + conv_len, sizeof(conv_prompt) - conv_len,
                                "<|im_start|>system\nYou are a helpful, brilliant AI assistant powered by Qualcomm Hexagon NPU.<|im_end|>\n");
        }
    }
}

// -------------------------------------------------------------
// OpenAI HTTP Server Mode (--server [port])
// -------------------------------------------------------------
typedef struct {
    int client_fd;
    int is_sse;
    char id[32];
    int chunk_count;
    char* full_resp;
    size_t full_cap;
    size_t full_len;
} HttpStreamContext;

static void http_token_cb(int tok_id, const char* tok_str, int is_final, void* user_arg) {
    HttpStreamContext* ctx = (HttpStreamContext*)user_arg;
    (void)tok_id;

    if (ctx->is_sse) {
        if (!is_final) {
            char esc[1024];
            json_escape(tok_str, (int)strlen(tok_str), esc, sizeof(esc));
            char sse_chunk[2048];
            int clen = snprintf(sse_chunk, sizeof(sse_chunk),
                                "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"bonsai-27b\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                                ctx->id, (long)time(NULL), esc);
            write(ctx->client_fd, sse_chunk, (size_t)clen);
            ctx->chunk_count++;
        }
    } else {
        if (!is_final) {
            size_t tlen = strlen(tok_str);
            if (ctx->full_len + tlen < ctx->full_cap) {
                memcpy(ctx->full_resp + ctx->full_len, tok_str, tlen);
                ctx->full_len += tlen;
                ctx->full_resp[ctx->full_len] = '\0';
            }
        }
    }
}

static void send_http_headers(int fd, int status, const char* content_type, int is_sse) {
    char hdr[512];
    int len = snprintf(hdr, sizeof(hdr),
                       "HTTP/1.1 %d OK\r\n"
                       "Content-Type: %s\r\n"
                       "Access-Control-Allow-Origin: *\r\n"
                       "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                       "Access-Control-Allow-Headers: *\r\n"
                       "%s"
                       "Connection: %s\r\n\r\n",
                       status, content_type,
                       is_sse ? "Cache-Control: no-cache\r\n" : "",
                       is_sse ? "keep-alive" : "close");
    write(fd, hdr, (size_t)len);
}

static void handle_http_client(int client_fd, float* logits_buf) {
    char req[65536];
    ssize_t n = read(client_fd, req, sizeof(req) - 1);
    if (n <= 0) { close(client_fd); return; }
    req[n] = '\0';

    // CORS preflight
    if (strncmp(req, "OPTIONS", 7) == 0) {
        const char* cors = "HTTP/1.1 204 No Content\r\n"
                           "Access-Control-Allow-Origin: *\r\n"
                           "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                           "Access-Control-Allow-Headers: *\r\n"
                           "Content-Length: 0\r\n\r\n";
        write(client_fd, cors, strlen(cors));
        close(client_fd);
        return;
    }

    // Health check
    if (strncmp(req, "GET /health", 11) == 0) {
        const char* resp = "{\"status\":\"ok\",\"model\":\"bonsai-27b\",\"backend\":\"Hexagon CDSP v79 HVX\"}\n";
        send_http_headers(client_fd, 200, "application/json", 0);
        write(client_fd, resp, strlen(resp));
        close(client_fd);
        return;
    }

    // Models list
    if (strncmp(req, "GET /v1/models", 14) == 0) {
        const char* resp = "{\"object\":\"list\",\"data\":[{\"id\":\"bonsai-27b\",\"object\":\"model\",\"created\":1712000000,\"owned_by\":\"custom\"}]}\n";
        send_http_headers(client_fd, 200, "application/json", 0);
        write(client_fd, resp, strlen(resp));
        close(client_fd);
        return;
    }

    // Chat completions endpoint
    if (strncmp(req, "POST /v1/chat/completions", 25) == 0) {
        const char* body = strstr(req, "\r\n\r\n");
        if (!body) body = strstr(req, "\n\n");
        if (!body) { close(client_fd); return; }
        body += (body[0] == '\r') ? 4 : 2;

        int stream = parse_json_bool(body, "stream", 1);
        int max_tokens = parse_json_int(body, "max_tokens", 512);
        if (max_tokens <= 0 || max_tokens > 4096) max_tokens = 512;

        ServerChatMessage msgs[64];
        int num_msgs = parse_chat_messages(body, msgs, 64);
        if (num_msgs <= 0) {
            const char* err = "{\"error\":{\"message\":\"No valid messages provided in request\",\"type\":\"invalid_request_error\"}}\n";
            send_http_headers(client_fd, 400, "application/json", 0);
            write(client_fd, err, strlen(err));
            close(client_fd);
            return;
        }

        char formatted_prompt[65536];
        format_bonsai_prompt(msgs, num_msgs, formatted_prompt, sizeof(formatted_prompt));
        free_chat_messages(msgs, num_msgs);

        HttpStreamContext ctx;
        ctx.client_fd = client_fd;
        ctx.is_sse = stream;
        ctx.chunk_count = 0;
        ctx.full_resp = NULL;
        ctx.full_cap = 0;
        ctx.full_len = 0;
        snprintf(ctx.id, sizeof(ctx.id), "chatcmpl-%ld", (long)time(NULL));

        if (stream) {
            send_http_headers(client_fd, 200, "text/event-stream", 1);
        } else {
            ctx.full_cap = 65536;
            ctx.full_resp = (char*)malloc(ctx.full_cap);
            if (ctx.full_resp) ctx.full_resp[0] = '\0';
        }

        int gen_tokens = generate_stream(formatted_prompt, max_tokens, logits_buf, http_token_cb, &ctx);

        if (stream) {
            char final_chunk[512];
            int flen = snprintf(final_chunk, sizeof(final_chunk),
                                "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"bonsai-27b\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                                "data: [DONE]\n\n",
                                ctx.id, (long)time(NULL));
            write(client_fd, final_chunk, (size_t)flen);
        } else {
            send_http_headers(client_fd, 200, "application/json", 0);
            char esc_resp[65536];
            json_escape(ctx.full_resp ? ctx.full_resp : "", (int)ctx.full_len, esc_resp, sizeof(esc_resp));
            char full_json[65536];
            int jlen = snprintf(full_json, sizeof(full_json),
                                "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"bonsai-27b\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":%d,\"total_tokens\":%d}}\n",
                                ctx.id, (long)time(NULL), esc_resp, gen_tokens, gen_tokens);
            write(client_fd, full_json, (size_t)jlen);
            if (ctx.full_resp) free(ctx.full_resp);
        }

        close(client_fd);
        return;
    }

    // Default 404
    const char* not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 13\r\n\r\n404 Not Found";
    write(client_fd, not_found, strlen(not_found));
    close(client_fd);
}

static void run_http_server(int port, float* logits_buf) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("[server] socket failed");
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons((uint16_t)port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("[server] bind failed");
        close(server_fd);
        return;
    }

    if (listen(server_fd, 16) < 0) {
        perror("[server] listen failed");
        close(server_fd);
        return;
    }

    printf("\n");
    printf("====================================================================\n");
    printf("  Bonsai 2 27B OpenAI HTTP Server Daemon Online!\n");
    printf("  Listening on: http://0.0.0.0:%d\n", port);
    printf("  Compatible with OpenWebUI, LibreChat, Chatbox, Cursor, and curl!\n");
    printf("  Endpoints:\n");
    printf("    POST http://localhost:%d/v1/chat/completions (SSE stream supported)\n", port);
    printf("    GET  http://localhost:%d/v1/models\n", port);
    printf("    GET  http://localhost:%d/health\n", port);
    printf("====================================================================\n\n");
    fflush(stdout);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            continue;
        }
        handle_http_client(client_fd, logits_buf);
    }

    close(server_fd);
}

#endif // BONSAI_SERVER_H
