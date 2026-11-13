"""Put a branch on main as one commit under the local git identity.

    python tools/squash-to-main.py --message-file MSG     stage: build the commit and show it; nothing is pushed
    python tools/squash-to-main.py --push                 push the staged commit to the remote's main

The staged commit's tree is the branch tip's tree as the remote has it, and its
parent is the remote's main, so main must be an ancestor of the branch (checked).
Its author and committer are git's user.name and user.email, and a message that
carries a Co-authored-by or Signed-off-by line is refused. No working tree is
touched and nothing is checked out; the commit waits under refs/staged/main.

--mode merge keeps the branch's commits behind a merge commit (GitHub then shows
the pull request as merged). The default, squash, puts one commit on main and
leaves the branch as the only holder of its history: do not delete the branch,
the calibration plan cites its hashes.

Pushing to main deploys the site. --push asks before it does.
"""
import argparse
import os
import re
import subprocess
import sys

STAGED = "refs/staged/main"


def git(*args, env=None):
    r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", env=env)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args[:2])} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def is_ancestor(a, b):
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b]).returncode == 0


def web_url(remote):
    """https://host/owner/repo for the remote, without any credentials in its URL."""
    url = git("remote", "get-url", remote)
    m = re.match(r"(?:https?://(?:[^@/]+@)?|git@|ssh://git@)([^/:]+)[/:](.+?)(?:\.git)?/?$", url)
    return f"https://{m.group(1)}/{m.group(2)}" if m else None


def identity():
    name, email = git("config", "user.name"), git("config", "user.email")
    if not name or not email:
        sys.exit("git's user.name and user.email must be set")
    return name, email


def describe(sha, remote):
    print(git("show", "-s", "--format=commit    %H%nauthor    %an <%ae>%ncommitter %cn <%ce>%nparents   %P%nsubject   %s", sha))
    print("change    " + git("diff", "--shortstat", f"{sha}^1", sha))
    base = web_url(remote)
    if base:
        print(f"link      {base}/commit/{sha}")


def stage(a):
    if not a.message_file:
        sys.exit("--message-file is needed to stage")
    with open(a.message_file, encoding="utf-8-sig") as f:
        message = f.read().strip() + "\n"
    if re.search(r"^\s*(co-authored-by|signed-off-by)\s*:", message, re.I | re.M):
        sys.exit("the message carries a Co-authored-by or Signed-off-by line; remove it")
    git("fetch", a.remote, a.main, a.branch)
    main, tip = git("rev-parse", f"{a.remote}/{a.main}"), git("rev-parse", f"{a.remote}/{a.branch}")
    if not is_ancestor(main, tip):
        sys.exit(f"{a.remote}/{a.main} is not an ancestor of {a.remote}/{a.branch}; merge main into the branch first")
    if main == tip:
        sys.exit("nothing to put on main")
    name, email = identity()
    env = dict(os.environ, GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email)
    env.pop("GIT_AUTHOR_DATE", None)
    env.pop("GIT_COMMITTER_DATE", None)
    parents = ["-p", main] + (["-p", tip] if a.mode == "merge" else [])
    r = subprocess.run(["git", "commit-tree", f"{tip}^{{tree}}", *parents], input=message, capture_output=True, text=True, encoding="utf-8", env=env)
    if r.returncode != 0:
        sys.exit(f"git commit-tree failed:\n{r.stderr.strip()}")
    sha = r.stdout.strip()
    git("update-ref", STAGED, sha)
    print(f"staged ({a.mode}) from {a.remote}/{a.branch} at {tip[:7]}; nothing is pushed\n")
    describe(sha, a.remote)
    print(f"\nto put it on main:  python tools/squash-to-main.py --push")


def push(a):
    sha = git("rev-parse", "--verify", STAGED)
    git("fetch", a.remote, a.main, a.branch)
    main, tip = git("rev-parse", f"{a.remote}/{a.main}"), git("rev-parse", f"{a.remote}/{a.branch}")
    if git("rev-parse", f"{sha}^1") != main:
        sys.exit(f"{a.remote}/{a.main} has moved since staging; stage again")
    if git("rev-parse", f"{sha}^{{tree}}") != git("rev-parse", f"{tip}^{{tree}}"):
        sys.exit(f"{a.remote}/{a.branch} has changed since staging; stage again")
    describe(sha, a.remote)
    if not a.yes:
        answer = input(f"\nPush this commit to {a.remote}/{a.main}? It deploys the site. Type yes: ")
        if answer.strip().lower() != "yes":
            sys.exit("not pushed")
    git("push", a.remote, f"{sha}:refs/heads/{a.main}")
    git("update-ref", "-d", STAGED)
    print(f"\npushed to {a.remote}/{a.main}")
    if len(git("rev-parse", f"{sha}^@").split()) == 1:
        print("GitHub does not mark a pull request as merged by a squash made outside it: close the pull request by hand.")
        print(f"Keep the branch {a.branch}: it holds the history.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--branch", default="next/calibration-ui")
    p.add_argument("--main", default="main")
    p.add_argument("--remote", default="origin")
    p.add_argument("--mode", choices=["squash", "merge"], default="squash")
    p.add_argument("--message-file")
    p.add_argument("--push", action="store_true")
    p.add_argument("--yes", action="store_true", help="push without asking")
    a = p.parse_args()
    push(a) if a.push else stage(a)
