import subprocess 
subprocess.Popen("$MDB_HOME/build/Release/bin/server_sparql $MDB_HOME/data/example-rdf-database", shell=True)
# subprocess.Popen("python3 ${MDB_HOME}/scripts/sparql_query.py ${MDB_HOME}/data/example-sparql-query.rq", shell=True)