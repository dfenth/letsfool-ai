# LetsFool.ai
This is a project looking at explainability in AI models (specifically image classifiers) and how we can use explainability to craft inputs that are incorrectly classified. The aim is to make a cool web app that people can experiment with. This is kind of a learning experience for me in developing web apps and deploying machine learning models in a *'production'* environment.

The running web app is available at [letsfoolai.cloud](https://letsfoolai.cloud), and a deep dive into the particulars of this project can be found in my [blog post](https://dfnt.xyz/projects/ml/2026/02/28/letsfoolai.html).

This project uses the [Google Cloud Run](https://cloud.google.com/run) service to host the app, which also has nice integrations with GitHub for CI/CD, streamlining the workflow.

## Running Locally
If you want to run this app locally, it can be built with [Docker](https://www.docker.com/):
```bash
docker build -t letsfool .
```
(Can just use `.` since our dockerfile is named `Dockerfile`)

We can run the docker container with:
```bash
docker run --rm -p 8080:8080 letsfool
```
To access the app running locally, you should be able to go to `localhost:8080`. We don't need to mount any external volumes because the dockerfile copies the contents of `letsfool` into the image during the build step.
