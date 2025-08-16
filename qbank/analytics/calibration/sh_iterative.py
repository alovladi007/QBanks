import argparse, json, psycopg2
from sh_core import load_pool, iterative_sh, upsert_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--exam", "--exam_code", dest="exam_code", required=True)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--len", "--test_len", dest="test_len", type=int, default=30)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--theta", "--theta_dist", dest="theta_dist", default="normal0,1")
    ap.add_argument("--floor", type=float, default=0.02)
    ap.add_argument("--ceil", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--topic_tau", type=str, default=None)
    ap.add_argument("--topic_weights", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    t_tau=json.loads(args.topic_tau) if args.topic_tau else None
    t_w=json.loads(args.topic_weights) if args.topic_weights else None
    conn=psycopg2.connect(args.dsn)
    pool=load_pool(conn,args.exam_code)
    if not pool: print("No items found"); return
    kmap,seen,hist=iterative_sh(pool,args.tau,args.n,args.test_len,args.iters,args.alpha,args.theta_dist,args.floor,args.ceil,args.seed,t_tau,t_w)
    print("history=",hist)
    if not args.dry_run:
        upsert_k(conn,kmap); print("upserted=",len(kmap))
    conn.close()
if __name__=="__main__": main()