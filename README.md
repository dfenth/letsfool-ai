# LetsFool.ai
This is a project looking at explainability in AI models (specifically image classifiers) and how we can use explainability to craft inputs that are incorrectly classified.

The aim is to make a cool web app that people can experiment with.

This is kind of a learning experience for me in developing web apps and deploying machine learning models in a 'production' environment.

Uses a conda environment `letsfool`.

So to set things up:
```bash
conda activate letsfool

uvicorn main:app --reload
```

If it appears the `uvicorn` run is trying to load things that aren't associated with this project, try hard resetting the cache `Ctrl+Shift+R`.

# Docker
Build with:
```bash
docker build -t letsfool .
```
(Can just use `.` since our dockerfile is named `Dockerfile`)

We can run the docker container with:
```bash
docker run --rm -v ./letsfool:/app -p 8000:8000 letsfool
```
Which mounts the `letsfool` directory as an external volume and gives access to port 8000 (maps docker's port 8000 to the local port 8000).

Doing the build and run every time is annoying. So let's [compose](https://docs.docker.com/compose/intro/features-uses/)! Compose not used since we don't have multiple containers because NGINX not required because Google Cloud Run acts as a managed reverse proxy (handles load balancing etc).

Gunicorn workers set to 1 so we don't have explosive worker count when spinning up multiple containers with Google Cloud Run.


Next:
- Sort out `main.py` which has an extension that should not be used in prod.
- Look at docker compose.
- Look at Kubernetes and maybe gunicorn for scaling.
- Add text to the page to explain what it is!
- Check this https://uvicorn.dev/settings/#production


Useful security tips from:
- https://snyk.io/blog/10-docker-image-security-best-practices/

Complete Docker reference:
- https://docs.docker.com/reference/dockerfile/
