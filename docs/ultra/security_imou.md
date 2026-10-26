# Security review: the Imou Open Platform integration

Scope: how BREE reaches the Truman Rd. DVR (serial AH040CBPAZ893C5) through the Imou Open Platform, using
`scripts/imou_record.py` (copy of the recorder, read-only methods only: `accessToken`, `deviceOnline`, `bindDeviceLive`,
`getLiveStreamInfo`). Evidence: code reading, plus one read-only structure check on 2026-10-25 (`accessToken` and one
`getLiveStreamInfo`, output reduced to scheme, host suffix, parameter names and a TLS handshake; no stream was pulled and no
value was printed). Rating: H high, M medium, L low.

## What the platform can see (M, and the part to tell the operator first)

* Imou's cloud is in the media path. Live video from the store DVR is relayed through Imou servers (`*.imoulife.com`) to us.
  Imou therefore can see the video, the device serial, which of our app ids asks for it, and our IP address.
* The API response carries a `coverUrl` per channel: a path under the device serial with a timestamp, i.e. Imou keeps
  **preview snapshots** of the store. We do not control their retention. The operator should be told, and may want the
  snapshot feature switched off in the Imou app (that is a device setting: not changed by us).
* Imou meters live streaming. The free flow ran out at about 19:09 on 2026-10-25 and every stream then returned a stub
  (error 11003, a 50 KB file). Availability of this path is a commercial dependency, not only a technical one.
* A permanent fix for privacy and availability is a LAN capture (RTSP/ONVIF from the DVR to the edge box). Then no video
  leaves the store. Cost: needs a visit and the DVR's local credentials.

## Findings

| # | Sev | Finding | Status |
|---|---|---|---|
| 1 | M | The HLS address is a bearer URL: the path itself authorises the stream (6 path segments, query holds only `source` and `proto`). Whoever has the URL can watch the channel until it expires. | accepted; short-lived by design. Never log it (done: `scrub`), never put it on a command line (finding 2). |
| 2 | M | The recorder runs `ffmpeg -i <url>`: the tokenised URL is in argv, readable by every local user through `/proc/<pid>/cmdline` and `ps`. `bree-trust secrets --env ~/.config/bree/imou.env` finds exactly this class (tested with a decoy process). | **fix ready, not applied to the running recorder**: `bree.trust.secrets.private_playlist(url, dir)` writes a mode-0600 one-line playlist and ffmpeg is started with that path (verified end to end against a local HLS server: rc 0, file produced). Apply with `-protocol_whitelist file,crypto,tcp,tls,https,http -allowed_extensions ALL`. |
| 3 | M | The platform offers both `http://...:8888` and `https://...:8890`. The recorder sorts https first, so it uses https. But if the https entry were missing it would silently take plaintext. | recommend: refuse non-https (one-line change in `live_url`). |
| 4 | M | ffmpeg TLS verification: ffmpeg 9.0.2 here verifies by default (`tls_verify` default true). Older builds (4.x on Ubuntu 22.04 / JetPack) default to **not** verifying. | pass `-tls_verify 1` explicitly in the recorder. The HLS host's certificate does validate against the system store (checked with Python's default context: TLS 1.2, DigiCert, expires 2026-11-22). |
| 5 | L | API calls use `urllib` with the default context: certificate and hostname verified (checked with `openssl s_client`: TLS 1.3, GoDaddy chain, verify OK). | ok |
| 6 | L | `appSecret` is read from `~/.config/bree/imou.env`, mode 0600 (checked; `bree-trust doctor` fails on looser modes or a symlink). It is held in memory by the recorder and never written to the log: URLs are scrubbed, the secret is replaced if it ever appears in an error. The repo contains none of the credential values (test `test_repo_has_no_secrets_and_no_imou_credentials` compares the tree with the real values on this machine). | ok |
| 7 | L | The access token is held in memory only, renewed an hour before expiry, never persisted. `test_imou.py` in the workspace repo writes nothing to disk. | ok |
| 8 | L | Request signing is HMAC-SHA256 over time, nonce and secret with a fresh nonce per call. Replays are limited by the platform's time window. | ok |
| 9 | M | Nothing stops a code change from calling a state-changing method. The recorder asserts the 4-method allow-list in `Imou.call`. Any new client should reuse that list. The device password is not used by the recorder at all. | ok; keep the assert. |
| 10 | L | The recorder's log (`recorder.log`) is mode 0600; `recorder_status.json` holds counters and timestamps only. | ok |
| 11 | M | Footage on disk is unencrypted. Disk theft or a shared account exposes it. | use an encrypted volume (LUKS/fscrypt) on the edge box; folders 0700 (checked by doctor). |
| 12 | L | The supervisor script and `pkill -f` patterns can kill the recorder by accident (it happened once). | use the systemd unit pattern in `deployment.md`. |

## Session lifecycle (operational, not a vulnerability)

On 2026-10-25 every channel returned a stale one-entry playlist from about 18:55 until the flow allowance ended (11003 at
about 19:09). Re-requesting addresses did not help. Treat live access through Imou as best-effort and plan every feature
to run from recorded files (as the rest of this run did).

## Onboarding other stations with only a serial and a password

The "serial + password" flow means the station gives Imou-level access to its device. Design notes for the platform track:
keep one app id per environment, store each station's device credentials in a 0600 file or the OS keyring (never in the
station YAML), log every bind/live call in the audit log (`credential_use`, no values), and show the operator which
channels are being pulled and where video is relayed. A station that does not accept the cloud relay needs the LAN path.
