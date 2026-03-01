cat << 'EOF' > /scrape_travis.py
import os, time, csv, requests, json
from tqdm import tqdm

TOKEN    = os.environ["TRAVIS_TOKEN"]
BASE_URL = "https://api.travis-ci.com"
HEADERS  = {
    "Travis-API-Version": "3",
    "Authorization":      f"token {TOKEN}",
    "Accept":             "application/json",
}

# Same Java projects as original TravisTorrent
PROJECTS = [
    "apache/commons-lang",
    "apache/commons-math",
    "apache/commons-io",
    "google/guava",
    "junit-team/junit4",
    "checkstyle/checkstyle",
    "square/retrofit",
    "square/okhttp",
    "ReactiveX/RxJava",
    "mockito/mockito",
    "spring-projects/spring-framework",
    "apache/maven",
    "slf4j/slf4j",
    "FasterXML/jackson-databind",
    "apache/httpcomponents-core",
]

def get_builds(repo_slug, limit=100, offset=0):
    slug = repo_slug.replace("/", "%2F")
    url  = f"{BASE_URL}/repo/{slug}/builds"
    params = {"limit": limit, "offset": offset,
              "sort_by": "finished_at:desc"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if r.status_code == 200:
        return r.json()
    return None

def get_log(job_id):
    url = f"{BASE_URL}/job/{job_id}/log.txt"
    r   = requests.get(url, headers=HEADERS, timeout=30)
    return r.text if r.status_code == 200 else ""

FIELDS = [
    "build_id", "build_number", "state", "duration",
    "started_at", "finished_at", "event_type",
    "gh_is_pr", "pr_number", "branch",
    "commit_sha", "commit_message",
    "gh_project_name",
    "job_id", "job_state", "job_duration",
    "tr_tests_run", "tr_tests_failed", "tr_tests_passed",
    "log_snippet",
]

import re
def parse_log(log_text):
    """Extract test counts from Maven Surefire output."""
    tests_run = tests_failed = tests_passed = 0
    pattern = re.compile(r"Tests run:\s*(\d+).*?Failures:\s*(\d+)", re.DOTALL)
    for m in pattern.finditer(log_text):
        tests_run    += int(m.group(1))
        tests_failed += int(m.group(2))
    tests_passed = tests_run - tests_failed
    # grab last 500 chars of log as snippet
    snippet = log_text[-500:].replace("\n", " ").replace(",", " ")
    return tests_run, tests_failed, tests_passed, snippet

with open("/data/travistorrent.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()

    for project in PROJECTS:
        print(f"\n→ Scraping {project}")
        offset = 0
        total_written = 0

        while total_written < 2000:   # max 2000 builds per project
            data = get_builds(project, limit=100, offset=offset)
            if not data or not data.get("builds"):
                break

            builds = data["builds"]
            for build in tqdm(builds, desc=f"  offset={offset}"):
                # only keep failed or passed builds
                if build.get("state") not in ("passed", "failed", "errored"):
                    continue

                jobs = build.get("jobs", [])
                for job in jobs:
                    job_id    = job.get("id")
                    log_text  = get_log(job_id) if build["state"] in ("failed","errored") else ""
                    tr, tf, tp, snippet = parse_log(log_text)
                    time.sleep(0.2)   # be polite to API

                    writer.writerow({
                        "build_id":         build.get("id"),
                        "build_number":     build.get("number"),
                        "state":            build.get("state"),
                        "duration":         build.get("duration"),
                        "started_at":       build.get("started_at"),
                        "finished_at":      build.get("finished_at"),
                        "event_type":       build.get("event_type"),
                        "gh_is_pr":         build.get("event_type") == "pull_request",
                        "pr_number":        build.get("pull_request_number"),
                        "branch":           build.get("branch", {}).get("name"),
                        "commit_sha":       build.get("commit", {}).get("sha"),
                        "commit_message":   build.get("commit", {}).get("message","")[:100],
                        "gh_project_name":  project,
                        "job_id":           job_id,
                        "job_state":        job.get("state"),
                        "job_duration":     job.get("duration"),
                        "tr_tests_run":     tr,
                        "tr_tests_failed":  tf,
                        "tr_tests_passed":  tp,
                        "log_snippet":      snippet,
                    })
                    total_written += 1

            offset += 100
            if len(builds) < 100:
                break
            time.sleep(1)

        print(f"  ✓ {total_written} builds written for {project}")

print("\nDone → /data/travistorrent.csv")
EOF

python /scrape_travis.py