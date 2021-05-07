
class Evaluator {
public:
    /*class members */
    vector<array<float, EMBEDDING_DIM>> embedding_table;    //embedding table
    vector<array<float, EMBEDDING_DIM>> qres;
    vector<TInfo> thrinfo;
    CustomBarrier cstart;
    CustomBarrier cend;
    
    steady_clock::time_point start;
    steady_clock::time_point end;
    size_t core_count;

    /* constructors */
    Evaluator(int core_count) : core_count(core_count) {
        //allocate space for thrinfo
        thrinfo.resize(core_count);
    }


    void build_embedding_table(int num_features) {
        //random initialization of embedding_table
        embedding_table.resize(num_features);
        for (int i=0; i < num_features; i++) {
            for (int j=0; j < EMBEDDING_DIM; j++)
                embedding_table[i][j] = 0.01 * i; //myRand();
        }
    }

    /* zero initialization of res vectors */
    void init(int qcount) {
        // for(size_t i=0; i < core_count; i++) {
        //     thrinfo[i].mcount = 0;
        //     thrinfo[i].time = 0;
        //     thrinfo[i].count = 0;
        // }
        if(qres.size() == 0)
            qres.resize(qcount);
        for(int i=0; i<qcount; i++) {
            for(int j=0; j< EMBEDDING_DIM; j++){
                qres[i][j] = 0.0f;
            }
        }
    }
    void setqbase(vector<vector<vector<int>>> &partitioned_query) {
        int accum = 0;
        for(size_t i=0; i<partitioned_query.size(); i++) {
            thrinfo[i].qbase = accum;
            accum += partitioned_query[i].size();
        }
    }
};


class Baseline: public Evaluator {
public:
    /*class members */
    int num_features;

    /* constructors */
    Baseline(int num_features_, int core_count_) : Evaluator(core_count_), num_features(num_features_) {}

    /* Baseline process */
    void inline reduce(const vector< vector<int> > &query, const int core_id) {
        size_t qlen = query.size();
        int myqbase = thrinfo[core_id].qbase;
        for(size_t i=0; i < qlen; i++) {
            size_t curqlen = query[i].size();
            for(size_t j=0; j<curqlen; j++) {
                for(int l=0; l < EMBEDDING_DIM; l++) {
                    qres[i + myqbase][l] += embedding_table[query[i][j]][l];
                }
            }
        }
    }
    void process(const vector< vector<int> > &query, const int core_id) { 
        // Prologue
        cstart.barrier_wait(core_count);
        if(core_id == 0) {
            start = steady_clock::now();
        }
        thrinfo[core_id].tstart = steady_clock::now();
        // Core
        reduce(query, core_id);
        // Epilogue
        thrinfo[core_id].tend = steady_clock::now();
        cend.barrier_wait(core_count);
        if(core_id == 0) {
            end = steady_clock::now();
        }
    }
};