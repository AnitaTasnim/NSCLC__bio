# Run this in Google Colab (or any computer with internet). It takes about 1 minute.
# 1) Upload FR_string_query_genes.txt to Colab (left panel -> Files -> Upload).
# 2) Run this cell. 3) Download string_edges.tsv and send it back.
import io, requests, time, pandas as pd

genes = [g.strip() for g in open("FR_string_query_genes.txt") if g.strip()]
api = "https://string-db.org/api"
# map our gene symbols to STRING identifiers (species 9606 = human)
r = requests.post(f"{api}/tsv/get_string_ids",
                  data={"identifiers": "\r".join(genes), "species": 9606, "limit": 1,
                        "echo_query": 1, "caller_identity": "tep_reliability_study"})
ids = pd.read_csv(io.StringIO(r.text), sep="\t")
ids = ids[["queryItem", "stringId", "preferredName"]].drop_duplicates("queryItem")
time.sleep(1)
# all interactions among these proteins with combined score >= 0.4 (as in the original analysis)
r = requests.post(f"{api}/tsv/network",
                  data={"identifiers": "\r".join(ids.stringId), "species": 9606, "required_score": 400,
                        "caller_identity": "tep_reliability_study"})
net = pd.read_csv(io.StringIO(r.text), sep="\t")
back = dict(zip(ids.stringId, ids.queryItem))
out = pd.DataFrame({"a": net.stringId_A.map(back), "b": net.stringId_B.map(back), "score": net.score})
out = out.dropna().drop_duplicates()
out.to_csv("string_edges.tsv", sep="\t", index=False)
ver = requests.get(f"{api}/json/version").json()
print(f"{len(ids)} of {len(genes)} genes mapped; {len(out)} edges; STRING version {ver}")
try:
    from google.colab import files
    files.download("string_edges.tsv")
except ImportError:
    pass
