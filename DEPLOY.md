# Deploying

The app is two pieces, hosted separately:

| Piece | What | Where |
|---|---|---|
| Frontend | `frontend/index.html` + `config.js`: a static page | **Vercel** |
| API | FastAPI + PyTorch models (`Dockerfile`) | **Hugging Face Spaces** (free CPU), or Render / Railway / Fly |

Why not all on Vercel? Vercel's Python functions are capped at 250 MB
unzipped and have no GPU. PyTorch + transformers + ultralytics are several
GB, so the model API can't run there. Vercel is great at serving the page,
though.

```
browser ──► https://<project>.vercel.app          (static page)
   │
   └─ fetch ─► https://<user>-pipespacing.hf.space/api/analyze   (Docker, CPU)
```

The label-review tool (`review.html`) is local-only. It edits `data/` on the
machine running the API, so the public deploy turns it off
(`ENABLE_REVIEW=0`) and the Vercel build leaves the page out.

---

## 1. API on Hugging Face Spaces

1. Create a Space at <https://huggingface.co/new-space>: name it
   `pipespacing`, choose **Docker → Blank**, hardware **CPU basic (free)**.
2. Push this repo's code to it. The Space needs its own `README.md` header,
   so swap that one file in:

   ```sh
   git clone https://huggingface.co/spaces/<user>/pipespacing hf-space
   git archive main | tar -x -C hf-space
   cp deploy/hf-space/README.md hf-space/README.md
   cd hf-space && git add -A && git commit -m "deploy" && git push
   ```

   (Pushing needs a Hugging Face access token with write scope as the password.)
3. In the Space's **Settings → Variables**, add:

   | Variable | Value |
   |---|---|
   | `ALLOWED_ORIGINS` | `https://<project>.vercel.app` |
   | `ALLOWED_ORIGIN_REGEX` | `https://<project>(-.*)?\.vercel\.app` (optional: Vercel preview URLs) |

4. Wait for the build (~10 min the first time: it bakes the model weights
   into the image). Check `https://<user>-pipespacing.hf.space/api/health`.

**Expect:** about 10–20 s per image on the free CPU with the zero-shot
Grounding DINO detector, and about 1–3 s once the fine-tuned
`models/pipe_yolov8n.pt` is committed (train it with `pipeline/train.py`,
then commit that one file; `.gitignore` already allows it). Free Spaces
sleep after a period of inactivity, and the first request after that takes a
minute. The page shows "api: waking up…" meanwhile.

### Other hosts

The same `Dockerfile` runs on Render, Railway, or Fly. They set `$PORT`,
which the container respects. Set the same `ALLOWED_ORIGINS` variable.
Give it at least 2 GB RAM.

## 2. Frontend on Vercel

1. <https://vercel.com/new> → import the GitHub repo.
2. Leave the framework preset as **Other**. `vercel.json` already sets the
   build command and output directory (`frontend`).
3. **Environment Variables** → add `PIPESPACING_API` =
   `https://<user>-pipespacing.hf.space` (no trailing slash needed).
4. Deploy. The build (`scripts/vercel-build.sh`) writes that URL into
   `config.js` and removes `review.html`. Without the variable the build still
   succeeds, and the page shows "api: not configured". Add the variable later
   and redeploy (Deployments → ⋯ → Redeploy).

Every push to `main` redeploys automatically. If you change the API URL,
redeploy so `config.js` is regenerated.

## Local development is unchanged

```sh
uvicorn backend.main:app --port 8010    # serves the page + API + review tool
```

`config.js` is empty in the repo, so the page calls the same origin.

## Environment variables (API)

| Variable | Default | Meaning |
|---|---|---|
| `ALLOWED_ORIGINS` | none | Comma-separated browser origins allowed to call the API (`*` = any) |
| `ALLOWED_ORIGIN_REGEX` | none | Same, as a regex |
| `ENABLE_REVIEW` | `1` (`0` in Docker) | Label-review endpoints + page |
| `MAX_UPLOAD_MB` | `15` | Reject larger uploads |
| `MAX_SIDE` | `1600` | Downscale long side before inference |
