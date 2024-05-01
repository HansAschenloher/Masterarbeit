# Masterarbeit

This project contains the code for my master's thesis.

This project is divided into two directories.

    - `results` contains the result tables of the hyperparameter optimisation
    - `src` contains the srccode for all experiments
    - `clearml` contains the docker compose file for the clearml server

A Jupyter Notebook configuration is provided and can be started with:
```bash
cd src
docker compose up
```

Most trainings where logd via [clearml](https://clear.ml/).
This can be run locally or remotely via the clearml/docker-compose.yml.
Note that the default secrets in docker-compose will need to be changed if the server is exposed to the Internet.
After this the client needs to be configured.
For this the `clearml` needs to bee installed.
So it is best to first install the requirements via 

```
pip install -r src/requirements.txt
```

After this the `cleaml-init` CLI tool should be available.
Then simly run `cleaml-init` and past the server configuration into it.
The Configuration can be found in the web UI of clearml (default [localhost:8080](http:localhost:8080))
Then go to `Settings>Workspace>Create new credentials`, and simply copy this information into `cleaml-init`.

Now everything should run as expected.
