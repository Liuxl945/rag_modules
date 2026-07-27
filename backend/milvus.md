docker run -d \  
  --name milvus-etcd \
  -p 2379:2379 \
  -v C:\Users\Lenovo\Downloads\all-in-rag-main\milvus_etcd:/etcd \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  -e ETCD_DATA_DIR=/etcd \
  -e ETCD_ADVERTISE_CLIENT_URLS=http://0.0.0.0:2379 \
  -e ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379 \
  milvusdb/etcd:latest


docker run -d --name milvus-etcd -p 2379:2379 -v C:\Users\Lenovo\Downloads\all-in-rag-main\milvus_etcd:/etcd -e ALLOW_NONE_AUTHENTICATION=yes -e ETCD_DATA_DIR=/etcd -e ETCD_ADVERTISE_CLIENT_URLS=http://0.0.0.0:2379 -e ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379 milvusdb/etcd:latest


docker run -d \
  --name milvus-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v C:\Users\Lenovo\Downloads\all-in-rag-main\minio:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio:latest \
  server /data --console-address :9001


docker run -d --name milvus-minio -p 9000:9000 -p 9001:9001 -v "C:\Users\Lenovo\Downloads\all-in-rag-main\minio:/data" -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin minio/minio:latest server /data --console-address :9001


docker run -d \
  --name milvus-standalone \
  --link milvus-etcd:etcd \
  --link milvus-minio:minio \
  -p 19530:19530 \
  -p 9091:9091 \
  -v C:\Users\Lenovo\Downloads\all-in-rag-main\milvus:/var/lib/milvus \
  -e ETCD_ENDPOINTS=etcd:2379 \
  -e MINIO_ADDRESS=minio:9000 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -e MILVUS_AUTHORIZATION_ENABLED=false \
  -e MILVUS_LOG_LEVEL=info \
  milvusdb/milvus:2.6-20260708-51749e6f
  milvus run standalone

docker run -d --name milvus-standalone --link milvus-etcd:etcd --link milvus-minio:minio -p 19530:19530 -p 9091:9091 -v C:\Users\Lenovo\Downloads\all-in-rag-main\milvus:/var/lib/milvus -e ETCD_ENDPOINTS=etcd:2379 -e MINIO_ADDRESS=minio:9000 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin -e MILVUS_AUTHORIZATION_ENABLED=false -e MILVUS_LOG_LEVEL=info milvusdb/milvus:2.6-20260708-51749e6f milvus run standalone



docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v C:\Users\Lenovo\Downloads\all-in-rag-main\data\C9/cypher:/import \
  -v C:\Users\Lenovo\Downloads\all-in-rag-main/neo4j/plugins:/plugins \ # 挂载插件目录
  -e NEO4J_AUTH=neo4j/all-in-rag \
  -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" \ # 取消 APOC 限制
  neo4j:latest

docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -v C:/Users/Lenovo/Downloads/all-in-rag-main/data/C9/cypher:/import -v C:/Users/Lenovo/Downloads/all-in-rag-main/neo4j/plugins:/plugins -e NEO4J_AUTH=neo4j/all-in-rag -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" neo4j:latest

docker run -d \
  --name neo4j-etc \
  -p 7474:7474 -p 17687:7687 \
  -v C:\Users\Lenovo\Downloads\all-in-rag-main\neo4j:/data_import \
  -v C:/Users/Lenovo/Downloads/all-in-rag-main/neo4j/plugins:/plugins
  -v C:\Users\Lenovo\Downloads\all-in-rag-main\data\C9\cypher:/cypher
  -v neo4j_data:/data \
  -e NEO4J_AUTH=neo4j/all-in-rag \
  -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" \ # 取消 APOC 限制
  neo4j:latest

docker run -d --name neo4j-etc -p 7474:7474 -p 17687:7687 -v C:/Users/Lenovo/Downloads/all-in-rag-main/neo4j:/data_import -v C:/Users/Lenovo/Downloads/all-in-rag-main/neo4j/plugins:/plugins -v C:/Users/Lenovo/Downloads/all-in-rag-main/data/C9/cypher:/cypher -v neo4j_data:/data -e NEO4J_AUTH=neo4j/all-in-rag -e NEO4J_dbms_security_procedures_unrestricted="apoc.*" neo4j:latest

docker exec -i neo4j-etc cypher-shell -u neo4j -p all-in-rag "RETURN apoc.version()"

docker exec -i neo4j-etc cypher-shell -u neo4j -p all-in-rag --database neo4j < C:\Users\Lenovo\Downloads\all-in-rag-main\data\C9\cypher\neo4j_import.cypher