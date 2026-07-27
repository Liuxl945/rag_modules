etcd 环境变量

ALLOW_NONE_AUTHENTICATION=yes
ETCD_DATA_DIR=/etcd
ETCD_ADVERTISE_CLIENT_URLS=http://0.0.0.0:2379
ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379

minio环境变量
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin






docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio-server \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio:latest server /data --console-address ":9001"


# neo4j

docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v /Users/lxl/Documents/代码/all-in-rag-main/data:/data_import \
  -v neo4j_data:/data \
  -v neo4j_plugins:/plugins \
  -e NEO4J_AUTH=neo4j/all-in-rag \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_use__neo4j__config=true \
  neo4j:latest

docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v /Users/lxl/Documents/代码/all-in-rag-main/data/C9/cypher:/import \
  -e NEO4J_AUTH=neo4j/all-in-rag \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" \
  neo4j:latest


docker exec -i neo4j cypher-shell -u neo4j -p all-in-rag --database neo4j < /Users/lxl/Documents/代码/all-in-rag-main/data/C9/cypher/neo4j_import.cypher

# milvus 启动教程
  docker run -d \
  --name milvus-etcd \
  -p 2379:2379 \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  -e ETCD_DATA_DIR=/etcd \
  -e ETCD_ADVERTISE_CLIENT_URLS=http://0.0.0.0:2379 \
  -e ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379 \
  milvusdb/etcd:latest

docker run -d \
  --name milvus-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v /Users/lxl/Documents/minio:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio:latest \
  server /data --console-address :9001


  docker run -d \
  --name milvus-standalone \
  --link milvus-etcd:etcd \
  --link milvus-minio:minio \
  -p 19530:19530 \
  -p 9091:9091 \
  -v /Users/lxl/Documents/milvus:/var/lib/milvus \
  -e ETCD_ENDPOINTS=etcd:2379 \
  -e MINIO_ADDRESS=minio:9000 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -e MILVUS_AUTHORIZATION_ENABLED=false \
  -e MILVUS_LOG_LEVEL=info \
  -e MILVUS_QUERYNODE_CACHE_SIZE=1024 \
  -e MILVUS_PROXY_HEALTHCHECK_TIMEOUT=30 \
  -e MILVUS_DATA_COORDINATOR_COMPACTION_ENABLE=true \
  milvusdb/milvus:v2.6.19 \
  milvus run standalone


