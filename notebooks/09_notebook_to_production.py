# Databricks notebook source

# MAGIC %md
# MAGIC # Module 09 — From Notebook to Production (Jobs + Databricks Asset Bundles)
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Explain why "copy the notebook to the prod workspace" fails, and what **infrastructure-as-code** fixes
# MAGIC > - Read a **Databricks Asset Bundle**: bundle, resources, variables, **targets** (dev / staging / prod)
# MAGIC > - Deploy and run the churn job in **your dev target**, and see how the same files promote to staging and prod
# MAGIC >
# MAGIC > ⏱ **Time:** 35 minutes  ·  Answers production question **#8**: *How do we move this from my notebook to production safely?*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏚 The legacy way
# MAGIC
# MAGIC Going to production means: export the notebook, import it into the production environment, find-and-replace `dev_db` with `prod_db`, set up the scheduler by clicking through a UI, and hope the cluster settings match. Staging is "mostly the same as prod, except for the bits someone changed by hand." Nobody can answer *"what exactly is deployed in prod, and who changed it?"* without logging in and looking around.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea: one recipe, three kitchens
# MAGIC
# MAGIC A restaurant chain doesn't let each branch improvise the menu. There is one **recipe book**, and each branch follows it with its own local suppliers. Change the recipe book, and every branch changes the same way.
# MAGIC
# MAGIC A **Databricks Asset Bundle (DAB)** is that recipe book for your project. It's a folder in Git with:
# MAGIC
# MAGIC - your **code**: the training and scoring scripts
# MAGIC - a **`databricks.yml`** describing the **resources** that code needs (jobs, schedules, compute environments, permissions, even models and experiments) as YAML
# MAGIC - **variables** for everything that differs between environments (catalog, schema, workspace)
# MAGIC - **targets**, the named environments (`dev`, `staging`, `prod`), each setting those variables differently
# MAGIC
# MAGIC `databricks bundle deploy -t <target>` makes the workspace match the files. The job definition, its schedule, and its Python environment are created or updated to match, the same way every time. Because it lives in Git, every change to production is a **reviewed pull request** with a history.
# MAGIC
# MAGIC Two details make it practical for teams:
# MAGIC
# MAGIC 1. **`mode: development`** prefixes everything you deploy with `[dev your_name]` and pauses schedules, so twenty attendees can deploy the same bundle to the same workspace without collisions.
# MAGIC 2. **`mode: production`** adds guard rails. It's meant to be deployed by **CI/CD as a service principal**, not from someone's laptop.
# MAGIC
# MAGIC The one caveat: a bundle describes *what* to deploy, but not *whether* the model is good enough. That decision still comes from evidence, which is why `train.py` contains the champion/challenger check from Module 07.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧭 What's in `bundle/`
# MAGIC
# MAGIC ```text
# MAGIC bundle/
# MAGIC ├── databricks.yml            # bundle name, variables, targets
# MAGIC ├── resources/
# MAGIC │   └── churn_job.yml         # the job: train_and_evaluate → batch_score
# MAGIC └── src/
# MAGIC     ├── train.py              # Modules 04 + 07 as one parameterised script
# MAGIC     └── batch_score.py        # score the latest snapshot with @champion
# MAGIC ```
# MAGIC
# MAGIC ### `databricks.yml`: variables and targets
# MAGIC
# MAGIC ```yaml
# MAGIC variables:
# MAGIC   catalog:
# MAGIC     default: ml_workshop
# MAGIC   schema:
# MAGIC     description: Schema inside the catalog
# MAGIC   model_name:
# MAGIC     default: churn_model
# MAGIC
# MAGIC targets:
# MAGIC   dev:
# MAGIC     mode: development
# MAGIC     default: true
# MAGIC     variables:
# MAGIC       schema: ${workspace.current_user.short_name}   # your personal schema from Module 00
# MAGIC   staging:
# MAGIC     variables:
# MAGIC       catalog: ml_workshop_staging
# MAGIC       schema: churn
# MAGIC   prod:
# MAGIC     mode: production
# MAGIC     variables:
# MAGIC       catalog: ml_workshop_prod
# MAGIC       schema: churn
# MAGIC ```
# MAGIC
# MAGIC **What to point out**
# MAGIC - **Same code, different data.** In dev the job reads and writes `ml_workshop.<you>`, which is exactly the schema you've used all day. In prod it reads and writes `ml_workshop_prod.churn`. Nobody runs find-and-replace.
# MAGIC - Targets can also point at **different workspaces** (`workspace.host`), which is common for regulated production environments.
# MAGIC
# MAGIC ### `resources/churn_job.yml`: the job
# MAGIC
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     churn_training:
# MAGIC       name: churn-training-${bundle.target}
# MAGIC       tasks:
# MAGIC         - task_key: train_and_evaluate
# MAGIC           environment_key: ml
# MAGIC           spark_python_task:
# MAGIC             python_file: ../src/train.py
# MAGIC             parameters: [--catalog=${var.catalog}, --schema=${var.schema}, ...]
# MAGIC         - task_key: batch_score
# MAGIC           depends_on: [{ task_key: train_and_evaluate }]
# MAGIC           ...
# MAGIC       environments:
# MAGIC         - environment_key: ml
# MAGIC           spec:
# MAGIC             environment_version: "4"
# MAGIC             dependencies: [mlflow>=3.1, scikit-learn==1.6.1]
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 6 ? * MON"
# MAGIC         timezone_id: Asia/Jakarta
# MAGIC ```
# MAGIC
# MAGIC **What to point out**
# MAGIC - Two tasks with a **dependency**: scoring only runs if training succeeds.
# MAGIC - `environments` replaces the `%pip install` cells. Dependencies are declared once, version-controlled, and run on **serverless** job compute.
# MAGIC - The schedule is a weekly **safety net**. The *primary* trigger is drift (Module 08, Cell 8).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC ### Step 1 — Look at the script before deploying it
# MAGIC
# MAGIC Open `bundle/src/train.py` in your Git folder. It's Modules 04 and 07 condensed: load `customer_features`, fit the pipeline, log and register it, set `@challenger`, then compare against `@champion` on the test split and promote if it's at least as good. Everything environment-specific arrives as an **argument**:
# MAGIC
# MAGIC ```py
# MAGIC p.add_argument("--catalog", required=True)
# MAGIC p.add_argument("--schema", required=True)
# MAGIC ...
# MAGIC model_name = f"{args.catalog}.{args.schema}.{args.model_name}"
# MAGIC ```
# MAGIC
# MAGIC **What to point out**
# MAGIC - No hard-coded catalog, schema, or path anywhere. This is what makes a script *deployable*.
# MAGIC - It's a plain `.py` file, so it's easy to code-review and unit-test. Notebooks work as job tasks too (`notebook_task`), with parameters passed through widgets.
# MAGIC
# MAGIC ### Step 2 — Deploy to your dev target
# MAGIC
# MAGIC **Option A: from the workspace (no install needed)**
# MAGIC
# MAGIC 1. In your Git folder, open `bundle/databricks.yml`.
# MAGIC 2. Click the **Deployments** (🚀 bundle) icon in the editor sidebar.
# MAGIC 3. Select target **dev** → **Deploy**.
# MAGIC 4. When it finishes, the panel lists the deployed job **`[dev your_name] churn-training-dev`**. Click **Run**.
# MAGIC
# MAGIC **Option B: from your laptop with the Databricks CLI**
# MAGIC
# MAGIC ```bash
# MAGIC cd bundle
# MAGIC databricks bundle validate -t dev          # checks YAML, resolves variables
# MAGIC databricks bundle deploy   -t dev          # creates/updates the job in the workspace
# MAGIC databricks bundle run      -t dev churn_training   # runs it and streams the status
# MAGIC ```
# MAGIC
# MAGIC **What to point out**
# MAGIC - `validate` catches typos and missing variables *before* anything changes.
# MAGIC - Run `deploy` twice: the second time nothing changes. Deploys are **idempotent**, and the files are the truth.
# MAGIC - Open **Jobs & Pipelines**: your job has the `[dev …]` prefix, a paused schedule, and tags `managed_by: asset-bundle`. Editing it in the UI shows a warning, because the source of truth is Git.
# MAGIC
# MAGIC ### Step 3 — Watch it run, then find it from a notebook
# MAGIC
# MAGIC Create a notebook `09_notebook_to_production`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
me = w.current_user.me().user_name

job = next(j for j in w.jobs.list()                      # dev mode renamed it "[dev you] churn-training-dev"
           if j.settings.name.endswith("churn-training-dev") and j.creator_user_name == me)
print(f"Job ID: {job.job_id}  ← use this as RETRAIN_JOB_ID in Module 08")
for run in w.jobs.list_runs(job_id=job.job_id, limit=3):
    print(run.run_id, run.state.life_cycle_state, run.state.result_state)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Jobs are API objects. This is how Module 08's drift check triggers a retrain: `w.jobs.run_now(job_id=…)`.
# MAGIC - Open the run in the UI. Each task shows its logs, duration, and the MLflow run it produced. In the model page, a new version appears with a `promotion_note` tag written by the job.
# MAGIC
# MAGIC ### Step 4 — Promotion to staging and prod (instructor walkthrough)
# MAGIC
# MAGIC Promotion is the **same command with a different target**, normally run by CI/CD, not by people:
# MAGIC
# MAGIC ```bash
# MAGIC databricks bundle deploy -t staging     # after the pull request is merged to main
# MAGIC databricks bundle deploy -t prod        # after staging checks pass, often with a manual approval
# MAGIC ```
# MAGIC
# MAGIC A minimal GitHub Actions workflow:
# MAGIC
# MAGIC ```yaml
# MAGIC # .github/workflows/deploy.yml
# MAGIC name: deploy-churn-bundle
# MAGIC on:
# MAGIC   push:
# MAGIC     branches: [main]
# MAGIC jobs:
# MAGIC   deploy:
# MAGIC     runs-on: ubuntu-latest
# MAGIC     environment: production                 # GitHub environment with required reviewers
# MAGIC     env:
# MAGIC       DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
# MAGIC       DATABRICKS_CLIENT_ID: ${{ secrets.SP_CLIENT_ID }}          # service principal (OAuth M2M)
# MAGIC       DATABRICKS_CLIENT_SECRET: ${{ secrets.SP_CLIENT_SECRET }}
# MAGIC     steps:
# MAGIC       - uses: actions/checkout@v4
# MAGIC       - uses: databricks/setup-cli@main
# MAGIC       - run: databricks bundle validate -t staging && databricks bundle deploy -t staging
# MAGIC         working-directory: bundle
# MAGIC       - run: databricks bundle deploy -t prod
# MAGIC         working-directory: bundle
# MAGIC ```
# MAGIC
# MAGIC **What to point out**
# MAGIC - Production is deployed by a **service principal**, a robot identity, so no person's credentials run production jobs, and access is revoked by disabling one identity.
# MAGIC - The pull request *is* the change record: who changed what, who approved it, and when it went live.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✍️ Your turn (5 min)
# MAGIC
# MAGIC The business wants scores **twice a week** (Monday and Thursday). Change the schedule in `resources/churn_job.yml` and redeploy to dev. What changed in the workspace?
# MAGIC
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC ```yaml
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 6 ? * MON,THU"
# MAGIC         timezone_id: Asia/Jakarta
# MAGIC ```
# MAGIC
# MAGIC Redeploy (UI **Deploy**, or `databricks bundle deploy -t dev`). The job's schedule updates in place and stays paused, because dev mode pauses schedules. In a real team this would be a one-line pull request, reviewed and then promoted to prod by CI/CD.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | Legacy | Databricks Asset Bundles |
# MAGIC |---|---|---|
# MAGIC | Moving to prod | Export/import notebooks, find-and-replace names | `bundle deploy -t prod`, same files, different variables |
# MAGIC | Environment differences | Undocumented manual tweaks | Explicit per-target variables in Git |
# MAGIC | Job definitions | Clicked together in a UI | Declared in YAML, reviewed in pull requests |
# MAGIC | Who deployed what | Unknown | Git history + CI/CD logs; prod runs as a service principal |
# MAGIC | Parallel developers | Overwrite each other's jobs | `mode: development` prefixes and isolates everything |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *Twenty attendees deploy the same bundle to the same workspace. Why don't their jobs overwrite each other?*
# MAGIC >
# MAGIC > The `dev` target uses `mode: development`, which prefixes every resource with `[dev <username>]` and points the `schema` variable at each person's own schema. Each attendee gets their own job, writing to their own tables and model.
# MAGIC
# MAGIC > *What's the difference between `train.py` running in dev and in prod?*
# MAGIC >
# MAGIC > Only the arguments: catalog, schema, and experiment come from the target's variables. The code, dependencies, and job structure are identical. That's the guarantee that what you tested is what runs in production.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 08 — Keep It Healthy | Next: Module 10 — Wrap-Up →
