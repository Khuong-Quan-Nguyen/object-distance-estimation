---
title: PipeSpacing API
emoji: 🔧
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Pipe detection + depth-based spacing API
---

# PipeSpacing API

Backend for the PipeSpacing pipe-rack inspection demo: detects pipes in a
photo, estimates depth, and measures spacing and clearance between
neighbouring pipes.

- `GET /api/health`
- `POST /api/analyze` (multipart `file`, optional `od_mm`, `mode`, `min_clearance_mm`, …)

Source and docs: see the GitHub repository.
