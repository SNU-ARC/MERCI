#include "utils.h"
#include "evaluator.h"

#define likely(x)      __builtin_expect(!!(x), 1) 
#define unlikely(x)    __builtin_expect(!!(x), 0) 

#define INTERVAL 1

// Too Large Buffer size -> Perf Degradation 
// Too Small Buffer size -> Perf Degradation 
// BUF_SIZE should be set to 1024 in case of dblp dataset
#define BUF_SIZE (512)
#define LBUF_SIZE (512)
#define BUF_FLUSH (0.5f)
// Cond #1 BUF_SIZE * (1-BUF_FLUSH) should be large enough to avoid overflow

struct GroupInfo{
    int offset;
    int cluster_size;
    int memoizationTable_base;
};

class MERCI: public Evaluator {
public:
    vector<array<float, EMBEDDING_DIM>> memoizationTable;          //not created for baseline
    vector<array<float, EMBEDDING_DIM>> oneTable;           //not created for baseline

    int num_partition;
    GroupInfo* groupInfo;
    uint8_t groupInfo_size;
    vector<int> num_clusters_per_group;
    int pcount_accum, lcount_accum;
    int emb_table_size;
    /* constructors */
    MERCI(int np, int emb_table_size_, int core_count, GroupInfo* groupInfo_, vector<int>& num_cluster_per_group_accum) : Evaluator(core_count), num_partition(np) {
        groupInfo_size = num_cluster_per_group_accum.size();
        groupInfo = new GroupInfo[groupInfo_size];
        groupInfo = groupInfo_;
        num_clusters_per_group.push_back(num_cluster_per_group_accum[0]);
        for(auto i=1; i<int(num_cluster_per_group_accum.size())-1; i++)
            num_clusters_per_group.push_back(num_cluster_per_group_accum[i]-num_cluster_per_group_accum[i-1]);
        assert(groupInfo_size <= 256);
        emb_table_size = emb_table_size_;
    }

    void build_memoization_table() {
        build_embedding_table(emb_table_size);
        uint64_t table_size = 0;
        int new_size = -1;
        if(groupInfo[groupInfo_size-2].cluster_size==1) // if there exist features that only belong to test queries (not in train queries)
            new_size = groupInfo_size-2;
        else
            new_size = groupInfo_size-1;

        for(auto i=0; i< new_size; i++)     // except for the last group of cluster size one and group with features only in test queries
            table_size += (pow(2, groupInfo[i].cluster_size)-1)*num_clusters_per_group[i];

        memoizationTable.resize(table_size);
        cout << "============================= Memoization Table INFO =============================" << endl;
        cout << "Group ID   " << "Base Offset     " << "First Feature ID     " << "Cluster Size    " << "# Clusters" << endl;
        auto base = 0;
        for(int i=0; i<new_size; i++){
            const auto cluster_size = int(groupInfo[i].cluster_size);
            const auto partition_table_size = pow(2, cluster_size)-1;
            printf("%-10d %-15d %-20d %-15d %-10d\n", i, base, groupInfo[i].offset, cluster_size, num_clusters_per_group[i]);

            #pragma omp parallel for
            for(int j=0; j<num_clusters_per_group[i]; j++){
                auto embedding_base = groupInfo[i].offset + j*cluster_size;
                for(int k=1; k<partition_table_size+1; k++){
                    array<float, EMBEDDING_DIM> value;
                    for(int t=0; t<EMBEDDING_DIM; t++)
                        value[t] = 0;
                    for(int idx=0; idx<cluster_size; idx++){
                        if((k >> idx)&1){
                            for(auto t=0; t<EMBEDDING_DIM; t++)
                                value[t] += embedding_table[embedding_base+idx][t];
                        }
                    }
                    assert(base+j*partition_table_size+k-1 < table_size);
		            for(int l=0; l<EMBEDDING_DIM; l++)
	                    memoizationTable[int(base+j*partition_table_size+k-1)][l] = value[l];
                }

            }
            groupInfo[i].memoizationTable_base = base;
            base += partition_table_size*num_clusters_per_group[i];
        }

        oneTable.resize(emb_table_size-groupInfo[new_size].offset);
        for(int i=0; i<(emb_table_size-groupInfo[new_size].offset); i++){
            for(int t=0; t<EMBEDDING_DIM; t++){
                oneTable[i][t] = embedding_table[groupInfo[new_size].offset+i][t];
	        }
        }
        int one = 1;
        printf("%-10d %-15d %-20d %-15d %-10d\n", new_size, base, groupInfo[new_size].offset, one, emb_table_size-groupInfo[new_size].offset);

        cout << "=================================================================================" << endl;

        cout << "Memoization Table Generation Done, total " << table_size+emb_table_size-groupInfo[new_size].offset << " entries\n\n" << endl;
    }

    // TODO: Does not support groups more than 32
    struct LocalGroupInfo {
        int offsets[32];
        int bases[32];
        int cluster_sizes[32];
    };
    struct ProcessingInfo {
        int query_base;
        int lcount;
        int pcount;
        size_t curqlen;
        size_t i;
        int next_offset;
        int cur_cluster_id;
        int cluster_table_size;
        int cur_offset;
        int memoizationTable_base;
        int cluster_size;
        int section_id;
        int qbit;
        int pad[3];
    };
    void inline setvals(struct ProcessingInfo &p, struct LocalGroupInfo &lgi, int id, int feature_id, int cluster_size) {
        int diff = feature_id - lgi.offsets[id];
        p.cur_cluster_id = diff / cluster_size;
        p.qbit = 1 << (diff % cluster_size);
        p.cluster_size = cluster_size;
        p.cluster_table_size = (1 << cluster_size) -1;
        p.cur_offset = lgi.offsets[id];
        p.next_offset = lgi.offsets[id+1];
        p.memoizationTable_base = lgi.bases[id];
        p.section_id = id;
    }
    void inline processKeys(const int *qarr, const int *karr, int count, int qbase) {
        for(int j=0; j<count; j++) {
            __builtin_prefetch(&memoizationTable[karr[j+2]][0], 0, 3);
            __builtin_prefetch(&memoizationTable[karr[j+2]][16], 0, 3);
            __builtin_prefetch(&memoizationTable[karr[j+2]][32], 0, 3);
            __builtin_prefetch(&memoizationTable[karr[j+2]][48], 0, 3);

            for(int k = 0; k < EMBEDDING_DIM; k++) {
                qres[qarr[j] + qbase][k] += memoizationTable[karr[j]][k];
            }
        }
    }

    void inline processKeys_one(const int *qarr, const int *karr, int count, int qbase) {
        for(int j=0; j<count; j++) {
            __builtin_prefetch(&oneTable[karr[j+2]][0], 0, 3);
            __builtin_prefetch(&oneTable[karr[j+2]][16], 0, 3);
            __builtin_prefetch(&oneTable[karr[j+2]][32], 0, 3);
            __builtin_prefetch(&oneTable[karr[j+2]][48], 0, 3);
            for(int k = 0; k < EMBEDDING_DIM; k++) {
                qres[qarr[j] + qbase][k] += oneTable[karr[j]][k];
            }
        }
    }
    void process(vector< vector<int> > &query, const int core_id) {
        // Prologue
        //ready += 1;
        int *loads = new int[LBUF_SIZE];
        int *qloads = new int[LBUF_SIZE];
        int *eloads = new int[LBUF_SIZE];
        int *pkeys = new int[BUF_SIZE];
        int *qkeys = new int[BUF_SIZE];
        int *eqkeys = new int[BUF_SIZE];
        memset(loads, 0, LBUF_SIZE*4);
        memset(qloads, 0, LBUF_SIZE*4);
        memset(pkeys, 0, BUF_SIZE*4);
        memset(qkeys, 0, BUF_SIZE*4);
        memset(eqkeys, 0, LBUF_SIZE*4);
        memset(eloads, 0, LBUF_SIZE*4);

        int SECTION_COUNT;
        int one_offset;
        if(groupInfo[groupInfo_size-2].cluster_size==1){
            SECTION_COUNT = groupInfo_size-2;
            one_offset = groupInfo[groupInfo_size-2].offset;
        }
        else{
            SECTION_COUNT = groupInfo_size-1;
            one_offset = groupInfo[groupInfo_size-1].offset;
        }
        // cout << "here" << core_id <<  endl;
        ProcessingInfo p={thrinfo[core_id].qbase, 0, 0,// query_base, lcount, pcount
            0, 0,  // curqlen, i
            -1, // next_offset
            -1, // cur_cluster_id
            -1, // cluster_table_size
            -1, -1, -1, -1, 0, 0, 0}; // cur_offset, memoizationTable_base, cluster_size  
        LocalGroupInfo localGroupInfo;
        assert((INTERVAL==1 && sizeof(localGroupInfo.offsets)==32*4) || (INTERVAL==4 && sizeof(localGroupInfo.offsets)==16*4));

        for(auto i=0; i<groupInfo_size; i++){
            localGroupInfo.offsets[i] = groupInfo[i].offset;
            localGroupInfo.bases[i] = groupInfo[i].memoizationTable_base;
            localGroupInfo.cluster_sizes[i] = groupInfo[i].cluster_size;
        }

        cstart.barrier_wait(core_count);
        if(core_id == 0) {
            start = steady_clock::now();
        }
        size_t qlen = query.size();
        thrinfo[core_id].tstart = steady_clock::now();
        for (p.i=0; p.i < qlen; p.i++) {
            p.curqlen = query[p.i].size();
            if(unlikely(p.curqlen == 1)){
                for(int k = 0; k < EMBEDDING_DIM; k++) {
                    qres[p.i][k] += embedding_table[query[p.i][0]][k];
                }
            }
            else{
                p.qbit = 0;
                int first_query = query[p.i][0];
                for(int k=0; k<SECTION_COUNT; k++) {
                    if(first_query < localGroupInfo.offsets[k+1]) {
                        setvals(p, localGroupInfo, k, first_query, localGroupInfo.cluster_sizes[k]);
                        break;
                    }
                }
                if (unlikely(p.qbit == 0)) {
                    int diff = first_query - one_offset;
                    qloads[p.lcount] = p.i;
                    loads[p.lcount++] = diff;
                    for(size_t k=1; k<p.curqlen; k++) {
                        int diff = query[p.i][k] - one_offset;
                        qloads[p.lcount] = p.i;
                        loads[p.lcount++] = diff;
                    }

                }
                else {
                    for (size_t j = 1; j < p.curqlen; j++) {
                        int current_query = query[p.i][j];
                        if(current_query >= one_offset) {
                            if(p.qbit != 0) {
                                qkeys[p.pcount] = p.i;
                                pkeys[p.pcount++] = p.memoizationTable_base + p.cur_cluster_id * p.cluster_table_size + p.qbit-1;
                                p.qbit = 0;
                            }
                            int diff = current_query - one_offset;
                            qloads[p.lcount] = p.i;
                            loads[p.lcount++] = diff;
                        }
                        else if(current_query < p.next_offset){
                            int diff = (current_query-p.cur_offset);
                            int c_id = diff / p.cluster_size;
                            int c_rem = diff % p.cluster_size;
                            if(c_id == p.cur_cluster_id){     // if cluster id is same, just update qbit
                                p.qbit |= (1<<c_rem);
                                // change = false;
                            }
                            else{   // if cluster id is not the same, update pkeys
                                qkeys[p.pcount] = p.i;
                                pkeys[p.pcount++] = p.memoizationTable_base+p.cur_cluster_id*p.cluster_table_size+p.qbit-1;
                                p.qbit = 1 << c_rem;
                                p.cur_cluster_id = c_id;
                            }
                        }
                        else{  // if query belongs to the set of different size of cluster
                            qkeys[p.pcount] = p.i;
                            pkeys[p.pcount++] = p.memoizationTable_base + p.cur_cluster_id * p.cluster_table_size + p.qbit-1;
                            p.qbit = 0;
                            for(int k=p.section_id+1; k<SECTION_COUNT; k++) {  
                                if(current_query < localGroupInfo.offsets[k+1]) {
                                    setvals(p, localGroupInfo, k, current_query, localGroupInfo.cluster_sizes[k]);
                                    break;
                                }
                            }
                        }
                    }
                    if(p.qbit != 0){
                        qkeys[p.pcount] = p.i;
                        pkeys[p.pcount++] = p.memoizationTable_base+p.cur_cluster_id*p.cluster_table_size + p.qbit-1;
                    }
                }
                if(unlikely(p.pcount > (BUF_SIZE * BUF_FLUSH))) {
                    processKeys(qkeys, pkeys, p.pcount, p.query_base);
                    p.pcount = 0;
                }
                if(unlikely(p.lcount > (LBUF_SIZE * BUF_FLUSH))) {
                    processKeys_one(qloads, loads, p.lcount, p.query_base);
                    p.lcount = 0;
                }
            }
        }
        processKeys(qkeys, pkeys, p.pcount, p.query_base);
        processKeys_one(qloads, loads,p.lcount, p.query_base);
        thrinfo[core_id].tend = steady_clock::now();
        cend.barrier_wait(core_count);
        if(core_id == 0) {
            end = steady_clock::now();
        }
    }

    
    int inline getKey(int i1, int i2) {
        return i1 * PARTITION_SIZE + (i2 - i1 - 1) - i1 * (i1+1)/2;
    }

    bitset<PARTITION_SIZE> getBitSet(vector<int> &query, int sid, int eid, int pid) {
        int base = pid * PARTITION_SIZE;
        bitset<PARTITION_SIZE> key;
        for (int i=sid; i <= eid; i++) {
            key.set(query[i] - base);
        }
        return key;
    }

};

int main(int argc, const char *argv[]) {

    if (argc % 2 == 0) {
        cout << "Usage: ./bin/eval_merci -d <dataset name> -p <# of partitions> --memory_ratio <ratio with regard to emb table> -c <core count> -r <repeat>" << endl;
        return -1;
    }
    /* static file I/O variables */
    string homeDir = getenv("HOME");                        // home directory
    string merciDir = homeDir + "/MERCI/data/";                  // MERCI directory (located in $HOME/MERCI)
    string datasetDir = merciDir + "6_evaluation_input/";   // input dataset directory

    /*file I/O variables */
    string datasetName;
    string testFileName;
    ifstream testFile;

    /* counter variables */
    int num_features = 0;                   // total number of features (train + test)
    size_t core_count = thread::hardware_concurrency();
    int repeat = 5;
    int num_partition = -1;             // total number of partitions
    float memory_ratio = 0;
    int emb_table_size = 0;                 //total length of embedding table
    /* helper variables */
    char read_sharp;                    // read #
    int num_groups;                     // number of group with different sizes

    /////////////////////////////////////////////////////////////////////////////  
    cout << "Eval Phase 0: Reading Command Line Arguments..." << endl << endl;
    /////////////////////////////////////////////////////////////////////////////

    /* parsing command line arguments */
    for (int i = 0; i < argc; i++) {
        string arg(argv[i]);

        if (arg.compare("--dataset") == 0 || arg.compare("-d") == 0) {
            datasetName = string(argv[i+1]);
            ++i; //skip next iteration
        }
        else if (arg.compare("--num_partition") == 0 || arg.compare("-p") == 0) {
            num_partition = stoi(argv[i+1]);
            ++i;
        }
        else if (arg.compare("--memory_ratio") == 0) {
            memory_ratio = stof(argv[i+1]);
            ++i;
        }
        else if (arg.compare("-c") == 0) {
            core_count = stoi(argv[i+1]);
            ++i;
        }
        else if (arg.compare("-r") == 0) {
            repeat = stoi(argv[i+1]);
            ++i;
        }
    }
    /* Error Checking */
    if (num_partition < 1) {
        cout << "ARG ERROR: Insufficient options, please check parameters" << endl;
        return -1;
    }
    
    cout << "=================== EVAL INFO ==================" << endl;
    cout << "Embedding Dimension    : " << EMBEDDING_DIM << endl;
    cout << "Partition Size         : " << PARTITION_SIZE << endl;    
    cout << "Debug                  : " << DEBUG << endl;
    cout << "Number of Cores Supported by Hardware : " << thread::hardware_concurrency() << endl;
    cout << "Number of Cores        : " << core_count << endl;
    cout << "===============================================" << endl << endl;    

    ///////////////////////////////////////////////////////////////////////////// 
    cout << "Eval Phase 1: Retrieving Test Queries..." << endl << endl;
    ///////////////////////////////////////////////////////////////////////////// 

    //Step 1: Open test.dat file
    testFileName = datasetDir + datasetName + "/partition_" + to_string(num_partition) + "/" + "test_" + to_string(memory_ratio) + "X_" + to_string(EMBEDDING_DIM) + "dim_" + to_string(PARTITION_SIZE) + ".dat";
    cout << "Reading test file from " << testFileName << endl << endl;
    testFile.open(testFileName, ifstream::in);
    if (testFile.fail()) {
        cout << "FILE ERROR: Could not open test.dat file" << endl;
        return -1;
    }

    GroupInfo* groupInfo;
    //Step 2: Read meta data
    testFile >> read_sharp;
    testFile >> num_features;
    testFile >> emb_table_size;
    testFile >> read_sharp;
    testFile >> num_groups;

    groupInfo = new GroupInfo[num_groups];
    vector<int> num_cluster_per_group_accum(num_groups, -1);
    for(auto i=0; i<num_groups; i++){
        testFile >> groupInfo[i].offset;
        testFile >> groupInfo[i].cluster_size;
        testFile >> num_cluster_per_group_accum[i];
    }

    cout << "============= PARTITION META INFO =============" << endl;
    cout << "# of Features (train + test)    : " << num_features << endl;
    cout << "# of Partitions              : " << num_partition << endl;
    cout << "Partition Size               : " << PARTITION_SIZE << endl;
    cout << "# of Embedding Table entries : " << emb_table_size << endl;
    cout << "===============================================" << endl << endl;    

    if(num_partition * PARTITION_SIZE < 0.75 * emb_table_size || num_partition * PARTITION_SIZE > 1.25 * emb_table_size) {
        cout << "Total # of items in the partition : " << num_partition  * PARTITION_SIZE << endl;
        cout << "# of Embedding Table Entries : " << emb_table_size << endl;
        cout << "Shouldn't partition size be " << emb_table_size / num_partition << "? \n";
        assert(false);
    }
    //Step 3: Read and store query(test) transactions
    QueryData qd(testFile);
    testFile.close(); //close file

    vector<thread> t;
    t.resize(core_count);
    qd.partition(core_count);
    cout << endl;

    ///////////////////////////////////////////////////////////////////////////// 
    cout << "Eval Phase 2: Generating Memoization Table..." << endl << endl;
    ///////////////////////////////////////////////////////////////////////////// 
    
    MERCI merci(num_partition, emb_table_size, core_count, groupInfo, num_cluster_per_group_accum);
    merci.build_memoization_table();

    cout << "Eval Phase 3: Running MERCI..." << endl << endl;
    eval<MERCI>(merci, qd, core_count, t, repeat, "MERCI");
    cout << endl;
    
    #if DEBUG
    double rsum = 0.0f;
    size_t qlen = qd.query.size();
    for(size_t i=0; i < qlen; i++) {
        size_t curqlen = qd.query[i].size();
        for(size_t j=0; j< curqlen; j++) {
            for(size_t l=0; l < EMBEDDING_DIM; l++)
                rsum += merci.embedding_table[qd.query[i][j]][l];
        }
    }
    cout << "Correct Value : " << rsum << "\n";
    #endif
    return 0;
}
