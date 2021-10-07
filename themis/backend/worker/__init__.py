"""
This subpackage contains part of Themis's backend.

The front end consists of the user-facing API and validation. The front end
launches the back end in an allocation, which does all of the work of actually
executing runs.

The backend consists of a number of worker scripts.

The worker scripts are designed to execute on compute nodes.
The prepper and finisher scripts are launched and managed by submitter scripts.
There can be multiple submitter scripts active in a single compute allocation
(i.e. each ensemble can have multiple submitters active at once),
and each of those submitters will have some number of preppers and finishers
active.
"""
