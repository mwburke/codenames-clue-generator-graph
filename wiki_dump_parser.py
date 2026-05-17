import gzip, re, csv, os

class WikiGraphBuilder:
    def __init__(self, page_dump, pagelinks_dump, redirect_dump, linktarget_dump, output_dir="neo4j/import"):
        self.page_dump = page_dump
        self.pagelinks_dump = pagelinks_dump
        self.redirect_dump = redirect_dump
        self.linktarget_dump = linktarget_dump
        self.output_dir = output_dir
        self.valid_pages = {} # id -> title
        self.redirects = {} # id -> target title
        self.link_targets = {} # lt_id -> title
        os.makedirs(self.output_dir, exist_ok=True)

    def is_valid_title(self, title):
        title_str = title.decode('utf-8', errors='ignore').replace('_', ' ')
        # Only keep reasonable titles to save memory
        return not (":" in title_str or title_str.startswith("List of") or len(title_str.split()) > 3), title_str

    def parse_pages(self):
        page_pattern = re.compile(b"\((\d+),(\d+),'([^']+)'")
        print("Parsing pages...")
        with gzip.open(self.page_dump, 'rb') as f:
            for line in f:
                if not line.startswith(b"INSERT INTO"): continue
                for match in page_pattern.finditer(line):
                    page_id, ns, title = match.groups()
                    if ns == b'0':
                        valid, title_str = self.is_valid_title(title)
                        if valid:
                            self.valid_pages[int(page_id)] = title_str

    def parse_redirects(self):
        # Redirect format is (rd_from,rd_namespace,rd_title)
        redirect_pattern = re.compile(b"\((\d+),(\d+),'([^']+)'")
        print("Parsing redirects...")
        if not os.path.exists(self.redirect_dump):
            print("No redirect dump found, skipping.")
            return
        with gzip.open(self.redirect_dump, 'rb') as f:
            for line in f:
                if not line.startswith(b"INSERT INTO"): continue
                for match in redirect_pattern.finditer(line):
                    from_id, ns, to_title = match.groups()
                    if ns == b'0':
                        self.redirects[int(from_id)] = to_title.decode('utf-8', errors='ignore').replace('_', ' ')

    def parse_linktargets(self):
        # Format is (lt_id,lt_namespace,lt_title)
        target_pattern = re.compile(b"\((\d+),(\d+),'([^']+)'")
        print("Parsing link targets...")
        if not os.path.exists(self.linktarget_dump):
            print("No linktarget dump found, skipping.")
            return
        with gzip.open(self.linktarget_dump, 'rb') as f:
            for line in f:
                if not line.startswith(b"INSERT INTO"): continue
                for match in target_pattern.finditer(line):
                    lt_id, ns, title = match.groups()
                    if ns == b'0':
                        valid, title_str = self.is_valid_title(title)
                        if valid:
                            self.link_targets[int(lt_id)] = title_str

    def parse_links_and_build_graph(self):
        print("Parsing links and building CSVs...")
        # pagelinks is now: (pl_from, pl_from_namespace, pl_target_id)
        link_pattern = re.compile(b"\((\d+),(\d+),(\d+)\)")
        
        nodes_csv = os.path.join(self.output_dir, 'nodes.csv')
        edges_csv = os.path.join(self.output_dir, 'edges.csv')
        
        # Write nodes first
        with open(nodes_csv, 'w', newline='', encoding='utf-8') as f_nodes:
            writer = csv.writer(f_nodes)
            writer.writerow(['title:ID', ':LABEL'])
            for title in set(self.valid_pages.values()):
                writer.writerow([title, 'Page'])

        # Process links
        with gzip.open(self.pagelinks_dump, 'rb') as f_in, open(edges_csv, 'w', newline='', encoding='utf-8') as f_edges:
            writer = csv.writer(f_edges)
            writer.writerow([':START_ID', ':END_ID'])
            for line in f_in:
                if not line.startswith(b"INSERT INTO"): continue
                for match in link_pattern.finditer(line):
                    from_id, ns, target_id = match.groups()
                    from_id_int = int(from_id)
                    target_id_int = int(target_id)
                    
                    if ns == b'0' and from_id_int in self.valid_pages and target_id_int in self.link_targets:
                        # Resolve redirect
                        source_title = self.valid_pages[from_id_int]
                        if from_id_int in self.redirects:
                            source_title = self.redirects[from_id_int]
                        
                        target_title = self.link_targets[target_id_int]
                        writer.writerow([source_title, target_title])

if __name__ == "__main__":
    builder = WikiGraphBuilder(
        "data/enwiki-latest-page.sql.gz", 
        "data/enwiki-latest-pagelinks.sql.gz", 
        "data/enwiki-latest-redirect.sql.gz",
        "data/enwiki-latest-linktarget.sql.gz"
    )
    builder.parse_pages()
    builder.parse_redirects()
    builder.parse_linktargets()
    builder.parse_links_and_build_graph()
    print("Done generating Neo4j CSVs.")
