"""
Neo4j Graph Database Module
Handles storage and querying of agricultural knowledge graph triplets
"""

import logging
import json
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

try:
    from neo4j import GraphDatabase, Session
except ImportError:
    GraphDatabase = None
    Session = None

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a graph node (entity)"""
    id: str
    label: str  # Entity type (e.g., "Crop", "Disease", "Nutrient")
    properties: Dict


@dataclass
class GraphRelationship:
    """Represents a graph relationship"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Optional[Dict] = None


class KnowledgeGraph:
    """
    Neo4j-based Knowledge Graph for agricultural domain
    Stores triplets and supports multi-hop queries
    """
    
    # Define valid entity types
    ENTITY_TYPES = {
        "CROP": "Agricultural crops",
        "DISEASE": "Crop diseases",
        "NUTRIENT": "Soil nutrients/fertilizers",
        "PEST": "Agricultural pests",
        "PRACTICE": "Agricultural practices",
        "CHEMICAL": "Pesticides/chemicals",
        "SYMPTOM": "Disease symptoms",
        "ENVIRONMENT": "Environmental factors"
    }
    
    # Define valid relationships
    RELATIONSHIP_TYPES = [
        "AFFECTS", "CAUSES", "PREVENTS", "TREATS",
        "RECOMMENDS", "CONTRAINDICATED_FOR", "OCCURS_IN",
        "REQUIRES", "IMPROVES", "REDUCES", "RELATED_TO"
    ]
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None
    ):
        """
        Initialize Knowledge Graph
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name (None for default)
        """
        self.uri = uri or settings.neo4j.uri
        self.username = username or settings.neo4j.username
        self.password = password or settings.neo4j.password
        self.database = database if database else None
        
        # Initialize driver
        self.driver = None
        self._connect()
        
        # Create indexes
        self._create_indexes()
        
        logger.info(f"Knowledge Graph initialized: {self.uri}/{database}")
    
    def _connect(self):
        """Connect to Neo4j database"""
        if GraphDatabase is None:
            raise ImportError(
                "neo4j is required for KnowledgeGraph. Install dependencies with "
                "`pip install -r requirements.txt`, or run `python local_demo.py` "
                "for the dependency-light demo."
            )

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            logger.info("Successfully connected to Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            logger.info(f"Make sure Neo4j is running at {self.uri}")
            raise
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            with self.driver.session(database=self.database) as session:
                # Create index on entity IDs
                session.run("""
                    CREATE INDEX entity_id IF NOT EXISTS 
                    FOR (n:Entity) ON (n.id)
                """)
                
                # Create index on entity names
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS 
                    FOR (n:Entity) ON (n.name)
                """)
                
                logger.info("Indexes created/verified")
                
        except Exception as e:
            logger.warning(f"Could not create indexes: {str(e)}")
    
    def add_triplet(
        self,
        subject: str,
        relation: str,
        obj: str,
        subject_type: str = "ENTITY",
        object_type: str = "ENTITY",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a triplet (subject-relation-object) to the graph
        
        Args:
            subject: Subject entity
            relation: Relationship type
            obj: Object entity
            subject_type: Type of subject entity
            object_type: Type of object entity
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        if relation not in self.RELATIONSHIP_TYPES:
            logger.warning(f"Invalid relationship type: {relation}")
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                # Create or update nodes
                session.run("""
                    MERGE (s:Entity {id: $subject})
                    SET s.name = $subject, s.type = $subject_type
                """, {
                    "subject": subject,
                    "subject_type": subject_type
                })
                
                session.run("""
                    MERGE (o:Entity {id: $object})
                    SET o.name = $object, o.type = $object_type
                """, {
                    "object": obj,
                    "object_type": object_type
                })
                
                # Create relationship
                rel_props = metadata or {}
                session.run(f"""
                    MATCH (s:Entity {{id: $subject}})
                    MATCH (o:Entity {{id: $object}})
                    MERGE (s)-[r:{relation}]->(o)
                    SET r.metadata = $metadata
                """, {
                    "subject": subject,
                    "object": obj,
                    "metadata": json.dumps(rel_props, ensure_ascii=False)
                })
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding triplet: {str(e)}")
            return False
    
    def add_triplets_batch(self, triplets: List[Dict]) -> int:
        """
        Add multiple triplets efficiently
        
        Args:
            triplets: List of triplet dictionaries
                     {subject, relation, object, [subject_type, object_type, metadata]}
            
        Returns:
            Number of triplets added successfully
        """
        added = 0
        
        for triplet in triplets:
            success = self.add_triplet(
                subject=triplet.get('subject'),
                relation=triplet.get('relation'),
                obj=triplet.get('object'),
                subject_type=triplet.get('subject_type', 'ENTITY'),
                object_type=triplet.get('object_type', 'ENTITY'),
                metadata=triplet.get('metadata')
            )
            
            if success:
                added += 1
        
        logger.info(f"Added {added}/{len(triplets)} triplets to graph")
        return added
    
    def query_triplets(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None,
        terms: Optional[List[str]] = None,
        limit: int = 25
    ) -> List[Dict]:
        """
        Query triplets by subject, relation, or object
        
        Args:
            subject: Filter by subject (partial match supported)
            relation: Filter by relationship type
            obj: Filter by object (partial match supported)
            
        Returns:
            List of matching triplets
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Build query
                where_clauses = []
                params = {}
                
                if subject:
                    where_clauses.append("toLower(s.name) CONTAINS toLower($subject)")
                    params['subject'] = subject
                
                if relation:
                    where_clauses.append(f"type(r) = $relation")
                    params['relation'] = relation
                
                if obj:
                    where_clauses.append("toLower(o.name) CONTAINS toLower($object)")
                    params['object'] = obj

                if terms:
                    where_clauses.append("""
                        any(term IN $terms WHERE
                            toLower(s.name) CONTAINS term OR
                            toLower(o.name) CONTAINS term OR
                            toLower(type(r)) CONTAINS term
                        )
                    """)
                    params['terms'] = [term.lower() for term in terms if term]
                
                where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
                
                query = f"""
                    MATCH (s:Entity)-[r]->(o:Entity)
                    WHERE {where_clause}
                    RETURN s.name as subject, type(r) as relation, o.name as object,
                           r.metadata as metadata
                    LIMIT $limit
                """
                params["limit"] = limit
                
                result = session.run(query, params)
                
                triplets = []
                for record in result:
                    triplet = {
                        'subject': record['subject'],
                        'relation': record['relation'],
                        'object': record['object'],
                        'metadata': record['metadata']
                    }
                    triplets.append(triplet)
                
                logger.info(f"Found {len(triplets)} triplets")
                return triplets
                
        except Exception as e:
            logger.error(f"Error querying triplets: {str(e)}")
            return []
    
    def get_subgraph(
        self,
        center_entity: str,
        max_hops: int = 2,
        relation_filter: Optional[List[str]] = None
    ) -> Dict:
        """
        Get subgraph around an entity (multi-hop neighborhood)
        
        Args:
            center_entity: Central entity
            max_hops: Maximum relationship hops
            relation_filter: Filter by specific relationship types
            
        Returns:
            Subgraph with nodes and edges
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Build relationship filter
                rel_filter = ""
                if relation_filter:
                    rel_types = "|".join(relation_filter)
                    rel_filter = f"[:{rel_types}*1..{max_hops}]"
                else:
                    rel_filter = f"[*1..{max_hops}]"
                
                query = f"""
                    MATCH (center:Entity)
                    WHERE toLower(center.name) CONTAINS toLower($entity)
                    MATCH p = (center)-{rel_filter}-(n)
                    WITH nodes(p) as nodes, relationships(p) as rels
                    RETURN nodes, rels
                    LIMIT 20
                """
                
                result = session.run(query, {"entity": center_entity})
                
                subgraph_nodes = set()
                subgraph_edges = []
                
                for record in result:
                    nodes = record['nodes']
                    rels = record['rels']
                    
                    # Collect nodes
                    for node in nodes:
                        subgraph_nodes.add((node['name'], node.get('type', 'ENTITY')))
                    
                    # Collect edges
                    for rel in rels:
                        start_node = rel.start_node
                        end_node = rel.end_node
                        subgraph_edges.append({
                            'source': start_node['name'],
                            'target': end_node['name'],
                            'type': rel.type
                        })
                
                subgraph = {
                    'center': center_entity,
                    'nodes': list(subgraph_nodes),
                    'edges': subgraph_edges,
                    'node_count': len(subgraph_nodes),
                    'edge_count': len(subgraph_edges)
                }
                
                logger.info(f"Retrieved subgraph: {subgraph['node_count']} nodes, {subgraph['edge_count']} edges")
                return subgraph
                
        except Exception as e:
            logger.error(f"Error retrieving subgraph: {str(e)}")
            return {}
    
    def find_relationships(
        self,
        source: str,
        target: str,
        max_hops: int = 3
    ) -> List[List[str]]:
        """
        Find all paths between two entities
        
        Args:
            source: Source entity
            target: Target entity
            max_hops: Maximum relationship hops
            
        Returns:
            List of paths (each path is a list of entities)
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = f"""
                    MATCH p = shortestPath(
                        (s:Entity {{name: $source}})-[*1..{max_hops}]->(t:Entity {{name: $target}})
                    )
                    RETURN [n.name for n in nodes(p)] as path
                    LIMIT 5
                """
                
                result = session.run(query, {
                    "source": source,
                    "target": target
                })
                
                paths = []
                for record in result:
                    path = record['path']
                    paths.append(path)
                
                logger.info(f"Found {len(paths)} paths between {source} and {target}")
                return paths
                
        except Exception as e:
            logger.warning(f"Error finding paths: {str(e)}")
            return []
    
    def get_neighbors(
        self,
        entity: str,
        hops: int = 1,
        relation_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get neighboring entities
        
        Args:
            entity: Central entity
            hops: Number of relationship hops
            relation_type: Filter by relationship type
            
        Returns:
            Dictionary mapping relationship types to neighbor entities
        """
        try:
            with self.driver.session(database=self.database) as session:
                rel_filter = f":{relation_type}" if relation_type else ""
                
                query = f"""
                    MATCH (center:Entity {{name: $entity}})
                    MATCH (center)-[r{rel_filter}*1..{hops}]-(neighbor)
                    UNWIND r as rel
                    RETURN type(rel) as rel_type, collect(distinct neighbor.name) as neighbors
                """
                
                result = session.run(query, {"entity": entity})
                
                neighbors = {}
                for record in result:
                    rel_type = record['rel_type']
                    neighbor_list = record['neighbors']
                    neighbors[rel_type] = neighbor_list
                
                logger.info(f"Found {sum(len(v) for v in neighbors.values())} neighbors of {entity}")
                return neighbors
                
        except Exception as e:
            logger.error(f"Error getting neighbors: {str(e)}")
            return {}
    
    def get_statistics(self) -> Dict:
        """
        Get graph statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Count nodes
                nodes_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                node_count = nodes_result.single()['count'] if nodes_result else 0
                
                # Count relationships
                rels_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rels_result.single()['count'] if rels_result else 0
                
                # Count entity types
                types_result = session.run("""
                    MATCH (n:Entity)
                    RETURN coalesce(n.type, 'ENTITY') as type, count(*) as count
                    ORDER BY count DESC
                """)
                
                type_distribution = {}
                for record in types_result:
                    type_distribution[record['type']] = record['count']
                
                # Count relationship types
                rel_types_result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(*) as count
                    ORDER BY count DESC
                """)
                
                rel_type_distribution = {}
                for record in rel_types_result:
                    rel_type_distribution[record['rel_type']] = record['count']
                
                stats = {
                    'total_nodes': node_count,
                    'total_relationships': rel_count,
                    'entity_type_distribution': type_distribution,
                    'relationship_type_distribution': rel_type_distribution
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def clear_graph(self) -> bool:
        """
        Clear all nodes and relationships from graph
        
        Returns:
            True if successful
        """
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            
            logger.info("Graph cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize KG (uncomment to use with running Neo4j)
    # kg = KnowledgeGraph()
    
    # # Add sample triplets
    # triplets = [
    #     {'subject': 'Urea', 'relation': 'CONTRAINDICATED_FOR', 'object': 'Fungal Blast'},
    #     {'subject': 'Fungal Blast', 'relation': 'OCCURS_IN', 'object': 'Rice'},
    #     {'subject': 'Drainage', 'relation': 'PREVENTS', 'object': 'Fungal Blast'},
    # ]
    # kg.add_triplets_batch(triplets)
    
    # # Query
    # results = kg.query_triplets(subject='Urea')
    # for r in results:
    #     print(f"{r['subject']} - {r['relation']} - {r['object']}")
    
    print("KnowledgeGraph module ready")
