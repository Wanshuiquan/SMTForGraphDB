import subprocess 

def launch_mql(path="$MDB_HOME/data/example-rdf-database"):
    subprocess.Popen(f"$MDB_HOME/build/Release/bin/server_mql {path}", shell=True)
# subprocess.Popen("python3 ${MDB_HOME}/scripts/sparql_query.py ${MDB_HOME}/data/example-sparql-query.rq", shell=True)


def build_db(data, db):
    subprocess.Popen(f"$MDB_HOME/build/Release/bin/create_db_mql {data} {db}",shell=True)


if __name__ == "__main__":
    launch_mql()