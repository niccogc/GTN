solve the following TODOs in a new branch of the repository

**First**

Aim TRACKING is broken. it doesnt run there is some bug in the code, I get the following error

```console
raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/zhome/6b/e/212868/GTN/experiments/trackers.py", line 25, in _with_retry
    return operation()
           ^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/experiments/trackers.py", line 156, in init_run
    self.run = Run(repo=repo, experiment=experiment_name, log_system_params=True)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/ext/exception_resistant.py", line 73, in wrapper
    _SafeModeConfig.exception_callback(e, func)
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/ext/exception_resistant.py", line 50, in reraise_exception
    raise e
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/ext/exception_resistant.py", line 71, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/sdk/run.py", line 850, in __init__
    super().__init__(run_hash, repo=repo, read_only=read_only, experiment=experiment, force_resume=force_resume)
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/sdk/run.py", line 272, in __init__
    super().__init__(run_hash, repo=repo, read_only=read_only, force_resume=force_resume)
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/sdk/base_run.py", line 38, in __init__
    self.repo = get_repo(repo)
                ^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/sdk/repo_utils.py", line 28, in get_repo
    repo = Repo.from_path(repo)
           ^^^^^^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/sdk/repo.py", line 218, in from_path
    repo = Repo(path, read_only=read_only, init=init)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/sdk/repo.py", line 125, in __init__
    self._client = Client(remote_path)
                   ^^^^^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/aim_auth.py", line 65, in _patched_client_init
    _original_client_init(self, remote_path)
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/ext/transport/client.py", line 61, in __init__
    self.connect()
  File "/zhome/6b/e/212868/GTN/aim_auth.py", line 124, in _patched_connect
    return _original_connect(self)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/ext/transport/utils.py", line 15, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/aim/ext/transport/client.py", line 158, in connect
    response_json = response.json()
                    ^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/.venv/lib/python3.12/site-packages/requests/models.py", line 980, in json
    raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)
requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/zhome/6b/e/212868/GTN/experiments/run_grid_search.py", line 514, in <module>
    main()
  File "/zhome/6b/e/212868/GTN/experiments/run_grid_search.py", line 422, in main
    tracker = create_tracker(
              ^^^^^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/experiments/trackers.py", line 253, in create_tracker
    aim_tracker = AIMTracker(
                  ^^^^^^^^^^^
  File "/zhome/6b/e/212868/GTN/experiments/trackers.py", line 162, in __init__
    _with_retry(init_run, "Failed to initialize AIM run")
  File "/zhome/6b/e/212868/GTN/experiments/trackers.py", line 33, in _with_retry
    raise TrackerError(f"{error_msg}: {e}") from e
experiments.trackers.TrackerError: Failed to initialize AIM run: Expecting value: line 1 column 1 (char 0)
```

**Second**
The tracking in AIm is done in a way that when AIM returns an error, it retry three times and then it stops, which is implemented.

The problem is that when i bsub < *.sh the job, when AIM fails, the experiment stops, but the bstat still show the experiment running, so it doesnt get killed by bsub.

We should investigate this and return the right signal for the script to be completely killed.

**Third**
Aim after a bit of tracking gives unauthorized request back.
Is the token passed in every aim calls? is all proper? why we get this? are batch submitting? you can read how aim is configured on the eratostene server with the files /etc/nixos/master/modules/homelabModules/aim/* 

we need to debug this.
give the print statements necessary for the debug

**Fourth**

Think about local logging of aim runs and then submitting to the server, in like .aim folder? would that be a possibility? how much spaces a run occupies? test it to verify.

do another branch with an implemented method both for local aim tracking, and then a script to submit the runs to a Repo.

