import http.server as srv
import json


OUTPUT_CONFIG_PATH = "/userdata/runtime.cfg"


def validate_json(j):
    if j.get("draw_blobs") is None:
        return False
    if j.get("thresholds") is None:
        return False

    try:
        thresholds = j["thresholds"]
        for thr in thresholds:
            assert(len(thr) == 6)
            assert(0 <= thr[0] <= thr[1] <= 100)
            assert(-128 <= thr[2] <= thr[3] <= 127)
            assert(-128 <= thr[4] <= thr[5] <= 127)

    except Exception as exc:
        print("ERROR validating json: ", exc)
        return False

    return True


def write_config(j):
    if not validate_json(j):
        print("Invalid json config")
        return

    with open(OUTPUT_CONFIG_PATH, "w") as file:
        file.write(f"draw_blobs {1 if j['draw_blobs'] else 0}\n")
        file.write(f"thr_cnt {len(j['thresholds'])}\n")
        for thr in j["thresholds"]:
            file.write(f"thr   {thr[0]} {thr[1]} {thr[2]} {thr[3]} {thr[4]} {thr[5]}\n")


class WWHandler(srv.BaseHTTPRequestHandler):
    def do_GET(self):
        print("do_GET")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def do_POST(self):
        print("do_POST")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            json_payload = json.loads(post_data.decode('utf-8'))
            print(json_payload)
            write_config(json_payload)
            
        except json.JSONDecodeError:
            self.send_response(400)

        except Exception:
            self.send_response(500)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')


def main():
    server = srv.HTTPServer(("", 8000), WWHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server is interrupted")
    
    server.server_close()
    print("RCJ WW server is stopped...")


if __name__ == "__main__":
    main()
