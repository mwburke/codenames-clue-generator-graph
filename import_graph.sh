#!/bin/bash
# Must be run while the Neo4j docker container is stopped
docker-compose down
rm -rf neo4j/data/databases/neo4j neo4j/data/transactions/neo4j
docker run --rm -v $(pwd)/neo4j/data:/data -v $(pwd)/neo4j/import:/var/lib/neo4j/import neo4j:5 \
  neo4j-admin database import full neo4j --overwrite-destination --skip-bad-relationships=true --bad-tolerance=100000000 --nodes=/var/lib/neo4j/import/nodes.csv --relationships=/var/lib/neo4j/import/edges.csv
docker-compose up -d
