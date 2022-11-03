.. _ensemble_manager_examples:

========
Examples
========
This page will cover a couple of very simple examples of Themis usage.

.. seealso::
  The :ref:`themis_quickstart` page, which covers general usage information.

Hello World
===========
In Themis's equivalent of the classic programming example, we'll run the program ``echo``
with a couple of different arguments. First, we'll create a 3-line script,
named ``echo_wrapper.sh``, that executes
``echo``. (For this trivial example, the script is not necessary---
we could have Themis execute ``echo`` directly---but it is a useful example.)

.. code-block:: bash

    #!/bin/bash

    echo %%message%%

The "%%message%%" is a template, and will be replaced with a particular value for each run.

Next, we'll need to decide whether to use Themis's Python interface or its
command-line interface (CLI). Compared to the Python interface, the CLI is much denser.
This can be nice when you know what you're doing, but it can also be harder to understand.

Themis Command-Line Interface Setup
-----------------------------------
For the command-line interface, we'll first need to put the messages for each run
into a CSV file, named ``params.csv``.

.. code-block:: none

  message,
  hello world,
  hola mundo,
  bonjour monde,
  buongiorno mondo,
  vale munde,

The top row in the CSV, "message", is a header row---it declares the label of the
column.

Now execution is as simple as two shell commands.

.. code-block:: none

    # create a new Themis ensemble, passing in the application first,
    # and then the CSV declaring the variables for each run
    # (and implicitly, the number of runs)
    $ themis create echo_wrapper.sh params.csv

    # start Themis
    $ themis execute-local

That's it! We'll do the same thing in Python before moving on to what the results look like.
If you aren't interested in Themis's Python interface, feel free to skip the next section.

Themis Python Setup
-------------------
We need to write a Python script that executes the ensemble. Let's name the script ``echo_driver.py``.
First we'll need to create the ``Run`` objects for our ensemble:

.. code-block:: python

  import themis

  # construct a list of arguments to the ``echo`` command
  echo_messages = [
    "hello world",
    "hola mundo",
    "bonjour monde",
    "buongiorno mondo",
    "vale munde",
  ]
  # construct a list of Run objects, creating a sample consisting of one variable:
  # the message string for ``echo``, named 'message' (note it matches the template in
  # ``echo_wrapper.sh``, declared as %%message%%).
  my_runs = [themis.Run(sample={"message": msg}) for msg in echo_messages]

Next, to construct the ``Themis`` object:

.. code-block:: python

  mgr = themis.Themis.create(
    application="echo",
    runs=my_runs,
  )

Lastly, start Themis locally:

.. code-block:: python

  mgr.execute_local()


Now the script is complete. A full version is below.

.. code-block:: python

    import themis


    def main():
        # construct a list of arguments to the ``echo`` command
        echo_messages = [
            "hello world",
            "hola mundo",
            "bonjour monde",
            "buongiorno mondo",
            "vale munde",
        ]
        # construct a list of Run objects, passing in ``echo`` arguments
        my_runs = [themis.Run(args=msg) for msg in echo_messages]
        mgr = themis.Themis.create(
          application="echo_wrapper.sh",
          runs=my_runs,
        )
        mgr.execute_local()


    if __name__ == '__main__':
        main()

With the setup complete, all we need to do now to start the ensemble is invoke the script
on the command line.

.. code-block:: none

  $ python echo_driver.py

Execution Results
-----------------
Let's assume now that we've launched the ensemble with either the Python or command-line interface to Themis.
Now we can check on the progress of the ensemble.
First, we'll check how many runs the ensemble manager has completed.

.. code-block:: none

    $ themis progress
    |███████████████████████████████████| 100.0% Complete (5/5)

Great---everything has finished. Now to confirm that the outputs are what we expect.
Each run should have been executed in its own directory. In each run directory,
a file named ``run.log`` should hold the output (stdout and stderr) from that run; also, a copy of
``echo_wrapper.sh`` should exist that has been created (with appropriate run-specific modifications)
from the template.

.. code-block:: none

    $ ls
    driver.py   runs
    $ ls runs
    1    2    3    4    5
    $ ls runs/1
    run.log     echo_wrapper.sh
    $ cat runs/1/run.log
    hello world
    $ cat runs/5/run.log
    vale munde
    $ cat runs/5/echo_wrapper.sh
    #!/bin/bash

    echo vale munde

Everything looks good. We can see that in run 5, ``echo_wrapper.sh`` was modified so that
"%%message%%" was replaced by "vale munde". This only happened because our samples had a variable
named "message" and its value for run 5 was "vale munde".

Augmented Hello World
=====================
In this example, we'll augment the original "Hello World" example to make it a little more interesting,
and also to show off some additional features. Here are the changes we'll implement:

#.  We'll keep the general idea of echoing a string, but we'll use an actual MPI application to do it.
#.  We'll have Themis use the Flux resource manager.
#.  We'll read in the samples for each run from a csv file.
#.  Instead of having our run directories be named like ``runs/####``, we'll name them by the language of the
    message that run will execute.

Setup
-----
Below is the C source code new application we will be executing, ``mpi_echo.c``.
All it does is echo an argument in parallel:

.. code-block:: c

    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv) {
        if (argc < 1) return 1;
        int i=0;

        // Initialize the MPI environment
        MPI_Init(NULL, NULL);

        // Get the number of processes
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        // print out command line arguments
        for (i = 1; i < argc; i++)
        {
            printf(argv[i]);
            printf(" ");
        }
        // Print out generic command-line message
        printf("from processor %s, rank %d out of %d processors\n",
               processor_name, world_rank, world_size);
        MPI_Finalize();
        return 0;
    }

Let's compile the program and put the executable in ``usr/workspace/$USER/ensemble_demo/mpi_echo``. Then
here is the batch script, named ``batch_script_wrapper.sh`` we'll use as a wrapper around ``mpi_echo``:

.. code-block:: bash

    #!/bin/bash

    # arbitrary commands go here...
    # the following line is just to demonstrate
    echo "batch script starting"

    flux mini run -n5 usr/workspace/$USER/ensemble_demo/mpi_echo %%message%%

    # arbitrary commands go here...
    # the following line is just to demonstrate
    echo "batch script done"

The samples for each run, in a csv file named ``samples.csv`` with a header row:

.. code-block:: none

  language,message
  english,hello world
  spanish,hola mundo
  french,bonjour monde
  italian,buongiorno mondo
  latin,vale munde

Execution with Themis Command-Line Interface
--------------------------------------------
To complete the example using Themis's command-line interface (CLI), we just need to
execute a few commands.

.. code-block:: none

    # allocate 5 MPI tasks to each `batch_script_wrapper.sh` (-n 5), use flux (--flux),
    # and name the run directories like "languages/spanish" and "languages/french".
    # The value of "{language}" will be replaced by the value of the "language" variable
    # for each run
    $ themis create batch_script_wrapper.sh samples.csv -n5 --flux -r"languages/{language}"

    # start themis inside an allocation of 3 nodes in the pdebug queue with a time limit of
    # 20 minutes and using the wbronze bank
    $ themis execute-alloc -N3 -ppdebug -bwbronze -t20

Execution with Themis Python Interface
--------------------------------------
To complete the example using Themis's Python interface, we need to first write a script
that calls on Themis, and then execute that script. Let's name the script ``mpi_echo_driver.py``.

.. code-block:: python

    import themis
    # import a function to convert the CSV into a list of dictionaries
    from ibis.composite_samples import parse_file


    def main():
        my_runs = [
            themis.Run(sample=sample, tasks=5)
            for sample in parse_file("samples.csv", "csv")
        ]
        mgr = themis.Themis.create(
          application="batch_script_wrapper.sh",
          runs=my_runs,
          run_dir_names="languages/{language}",
          use_flux=True,
        )
        # request a 3-node allocation for 20 minutes in partition pdebug
        # charge the wbronze bank for the allocation
        alloc = themis.allocation(nodes=3, partition="pdebug", bank="wbronze", timeout=20)
        job_id = mgr.execute_alloc(alloc)
        print("Batch job ID is " + str(job_id))


    if __name__ == '__main__':
        main()


Results
-------
By invoking the Python script or executing the shell commands (with the CLI)
and waiting for the ensemble to complete, we get something
like the following. Note that instead of directories ``runs/####``
we get directories like ``languages/italian``,
thanks to the ``"languages/{language}"`` argument we specified.

.. code-block:: none

    [execute Themis Python script or CLI commands]
    Batch job ID is 58624
    $ ls
    batch_script_wrapper.sh  mpi_echo_driver.py  mpi_echo  mpi_echo.c  samples.csv
    [wait for the ensemble to complete...]
    $ ls
    batch_script_wrapper.sh  mpi_echo_driver.py  languages  mpi_echo  mpi_echo.c  samples.csv
    $ ls languages/
    english  french  italian  latin  spanish
    $ cat languages/latin/run.log
    batch script starting
    vale munde from processor rztopaz7, rank 0 out of 5 processors
    vale munde from processor rztopaz7, rank 2 out of 5 processors
    vale munde from processor rztopaz7, rank 3 out of 5 processors
    vale munde from processor rztopaz7, rank 1 out of 5 processors
    vale munde from processor rztopaz7, rank 4 out of 5 processors
    batch script done
    $ cat languages/latin/batch_script_wrapper.sh | grep flux
    flux mini run -n5 usr/workspace/$USER/ensemble_demo/mpi_echo vale munde

Comments
--------
*   There wasn't really any need to involve the ``batch_script_wrapper.sh`` file. Since all it did was execute
    ``mpi_echo``, we could have instead just told the ensemble manager to execute ``mpi_echo`` directly. However, if we were
    going to put additional commands or logic into ``batch_script_wrapper.sh``, then it would be more useful.
*   Instead of executing ``flux mini run -n5 mpi_echo %%message%%`` in ``batch_script_wrapper.sh``,
    we could have instead executed ``flux mini run -n5 mpi_echo $1``, and passed command-line arguments
    to the batch script. (``$1`` in Bash refers to the first command-line argument.)
    Either approach is valid.
