# History rewrite if institutional review requires it

The technical audit found no credentials or row-level WRDS/OptionMetrics records in the visible Git history. It did find a previously public detailed empirical report and generated aggregate fragments.

Do not rewrite history merely for cosmetic cleanup. A history rewrite is disruptive and does not remove copies already cloned, cached, forked, or downloaded.

Rewrite history only if Bocconi, WRDS, or OptionMetrics determines that the earlier derived-report publication was not permitted. In that event:

1. make the repository temporarily private;
2. preserve a private evidence copy for institutional counsel;
3. use `git filter-repo` to remove the identified paths from every ref;
4. force-push all rewritten branches and tags;
5. delete affected releases and GitHub Pages artifacts;
6. request cache removal where appropriate;
7. notify known collaborators to reclone; and
8. document the remediation and written authorization.
