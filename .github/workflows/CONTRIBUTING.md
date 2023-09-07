# Contribution Guide

This guide is intended for developers or administrators who want to contribute a new package, feature, or bugfix to Themis.
It assumes that you have at least some familiarity with Git VCS and Github.
The guide will show a few examples of contributing workflows and discuss the granularity of pull-requests (PRs). It will also discuss the tests your PR must pass in order to be accepted into Themis.

First, what is a PR? Quoting [Bitbucket's tutorials](https://www.atlassian.com/git/tutorials/making-a-pull-request/):

> Pull requests are a mechanism for a developer to notify team members that they have **completed a feature**. The pull request is more than just a notificationbitbs a dedicated forum for discussing the proposed feature.

Important is **completed feature**. The changes one proposes in a PR should correspond to one feature/bugfix/extension/etc. One can create PRs with changes relevant to different ideas, however reviewing such PRs becomes tedious and error prone. If possible, try to follow **one-PR-one-package/feature** rule.

## Branches

Themis' ``develop`` branch has the latest contributions. Nearly all pull requests should start from ``develop`` and target ``develop``.

## Continuous Integration

Themis uses [Github Actions](https://docs.github.com/en/actions) for Continuous Integration testing. This means that every time you submit a pull request, a series of tests will
be run to make sure you didn't accidentally introduce any bugs into Themis. **Your PR will not be accepted until it passes all of these tests.** While you can certainly wait
for the results of these tests after submitting a PR, we recommend that you run them locally to speed up the review process.

> **_NOTE_:** Oftentimes, CI will fail for reasons other than a problem with your PR. For example, apt-get, pip, or homebrew will fail to download one of the dependencies for the test suite, or a transient bug will cause the unit tests to timeout. If any job fails, click the "Details" link and click on the test(s) that is failing. If it doesn't look like it is failing for reasons related to your PR, you have two options. If you have write permissions for the Themis repository, you should see a "Restart workflow" button on the right-hand side. If not, you can close and reopen your PR to rerun all of the tests. If the same test keeps failing, there may be a problem with your PR. If you notice that every recent PR is failing with the same error message, it may be that an issue occurred with the CI infrastructure or one of Themis' dependencies put out a new release that is causing problems. If this is the case, please file an issue.

## Unit Tests

Unit tests ensure that core Themis features are working as expected. If your PR only adds new packages or modifies existing ones, there's very little chance that your changes could cause the unit tests to fail. However, if you make changes to Themis' core packages, you should run the unit tests to make sure you didn't break anything.

Make sure test dependencies are installed on your system and can be found in your ``PATH``. All of these can be installed with Themis.

To run *all* of the unit tests, use:

```bash 
$ pytest tests/
```

If you know you are only modifying a single Themis feature, you can run subsets of tests at a time.  For example, this would run all the tests in ``tests/test_plots``:

```bash 
$ pytest tests/unit/test_allocator.py
```

And this would run the ``ShellScriptTests.test_sleep`` test from that file:

```bash 
$ pytest tests/unit/test_allocator.py::ShellScriptTests::test_sleep
```

This allows you to develop iteratively: make a change, test that change, make another change, test that change, etc.  We use [pytest](http://pytest.org/) as our tests framework, and these types of arguments are just passed to the ``pytest`` command underneath. See the
[pytest docs](https://doc.pytest.org/en/latest/how-to/usage.html#specifying-which-tests-to-run) for more details on test selection syntax.


## Git Workflows

Themis is still in the beta stages of development. Most of our users run off of the develop branch, and fixes and new features are constantly being merged. So how do you keep up-to-date with upstream while maintaining your own local
differences and contributing PRs to Themis?

### Branching

The easiest way to contribute a pull request is to make all of your changes on new branches. Make sure your ``develop`` is up-to-date and create a new branch off of it:

```bash 
$ git checkout develop
$ git pull upstream develop
$ git branch <descriptive_branch_name>
$ git checkout <descriptive_branch_name>
```

Here we assume that the local ``develop`` branch tracks the upstream develop branch of ThemisS. This is not a requirement and you could also do the same with remote branches. But for some it is more convenient to have a local branch that
tracks upstream.

Normally we prefer that commits pertaining to a package ``<package-name>`` have a message ``<package-name>: descriptive message``. It is important to add
descriptive message so that others, who might be looking at your changes later (in a year or maybe two), would understand the rationale behind them.

Now, you can make your changes while keeping the ``develop`` branch pure. Edit a few files and commit them by running:

``` bash
$ git add <files_to_be_part_of_the_commit>
$ git commit --message <descriptive_message_of_this_particular_commit>
```

Next, push it to your remote fork and create a PR:

``` bash
$ git push origin <descriptive_branch_name> --set-upstream
```

GitHub provides a [tutorial](https://help.github.com/articles/about-pull-requests/) on how to file a pull request. When you send the request, make ``develop`` the destination branch.

If you need this change immediately and don't have time to wait for your PR to be merged, you can always work on this branch. But if you have multiple PRs, another option is to maintain a Frankenstein branch that combines all of your other branches:

``` bash
$ git checkout develop
$ git branch <your_modified_develop_branch>
$ git checkout <your_modified_develop_branch>
$ git merge <descriptive_branch_name>
```

This can be done with each new PR you submit. Just make sure to keep this local branch up-to-date with upstream ``develop`` too.


### Cherry-Picking

What if you made some changes to your local modified develop branch and already committed them, but later decided to contribute them to Themis? You can use cherry-picking to create a new branch with only these commits.

First, check out your local modified develop branch:

```bash 
$ git checkout <your_modified_develop_branch>
```

Now, get the hashes of the commits you want from the output of:

```bash
$ git log
```

Next, create a new branch off of upstream ``develop`` and copy the commits that you want in your PR:

```bash 
$ git checkout develop
$ git pull upstream develop
$ git branch <descriptive_branch_name>
$ git checkout <descriptive_branch_name>
$ git cherry-pick <hash>
$ git push origin <descriptive_branch_name> --set-upstream
```

Now you can create a PR from the web-interface of GitHub. The net result is as follows:

- You patched your local version of Themis and can use it further.
- You "cherry-picked" these changes in a stand-alone branch and submitted it as a PR upstream.

Should you have several commits to contribute, you could follow the same procedure by getting hashes of all of them and cherry-picking to the PR branch.

> **_NOTE_**: It is important that whenever you change something that might be of importance upstream, create a pull request as soon as possible. Do not wait for weeks/months to do this, because you might forget why you modified certain files or it could get difficult to isolate this change into a stand-alone clean PR.


### Rebasing

Other developers are constantly making contributions to Themis, possibly on the same files that your PR changed. If their PR is merged before yours, it can create a merge conflict. This means that your PR can no longer be automatically merged without a chance of breaking your changes. In this case, you will be asked to rebase on top of the latest upstream ``develop``.

First, make sure your develop branch is up-to-date:

```bash
$ git checkout develop
$ git pull upstream develop
```

Now, we need to switch to the branch you submitted for your PR and rebase it on top of develop:

```bash 
$ git checkout <descriptive_branch_name>
$ git rebase develop
```

Git will likely ask you to resolve conflicts. Edit the file that it says can't be merged automatically and resolve the conflict. Then, run:

```bash 
$ git add <file_that_could_not_be_merged>
$ git rebase --continue
```
You may have to repeat this process multiple times until all conflicts are resolved. Once this is done, simply force push your rebased branch to your remote fork:

```bash 
$ git push --force origin <descriptive_branch_name>
```

### Rebasing with cherry-pick

You can also perform a rebase using ``cherry-pick``. First, create a temporary backup branch:

```bash
$ git checkout <descriptive_branch_name>
$ git branch tmp
```

If anything goes wrong, you can always go back to your ``tmp`` branch. Now, look at the logs and save the hashes of any commits you would like to keep:

```bash
$ git log
```

Next, go back to the original branch and reset it to ``develop``. Before doing so, make sure that you local ``develop`` branch is up-to-date with upstream:

```bash
$ git checkout develop
$ git pull upstream develop
$ git checkout <descriptive_branch_name>
$ git reset --hard develop
```

Now you can cherry-pick relevant commits:

```bash
$ git cherry-pick <hash1>
$ git cherry-pick <hash2>
```

Push the modified branch to your fork:

```bash
$ git push --force origin <descriptive_branch_name>
```

If everything looks good, delete the backup branch:

```bash
$ git branch --delete --force tmp
```

### Re-writing History


Sometimes you may end up on a branch that has diverged so much from develop that it cannot easily be rebased. If the current commits history is more of an experimental nature and only the net result is important, you may rewrite
the history.

First, merge upstream ``develop`` and reset you branch to it. On the branch in question, run:

```bash
$ git merge develop
$ git reset develop
```

At this point your branch will point to the same commit as develop and thereby the two are indistinguishable. However, all the files that were previously modified will stay as such. In other words, you do not lose the changes you made. Changes can be reviewed by looking at diffs:

```bash
$ git status
$ git diff
```

The next step is to rewrite the history by adding files and creating commits:

```bash
$ git add <files_to_be_part_of_commit>
$ git commit --message <descriptive_message>
```
After all changed files are committed, you can push the branch to your fork and create a PR:

```bash
$ git push origin --set-upstream
```
