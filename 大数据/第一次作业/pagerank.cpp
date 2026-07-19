// ======= 可配置参数 =======
#define BLOCK_NUM 8 // 分块数量
#define TELEPORT_ALPHA 0.85 // teleport参数
#define EPSILON 1e-6 // 收敛精度
#define MAX_ITER 100 // 最大迭代次数
#define _CRT_SECURE_NO_WARNINGS
// PageRank分块外存优化版，所有代码均在本文件，含中文注释
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
using namespace std;

// 全局节点集合，保证所有遍历、初始化、归一化、输出都只针对实际出现的节点
unordered_set<int> all_nodes;
// 预处理：将Data.txt分块写入block1, block2, ...
void preprocess_blocks(const string& filename, int block_num, int& node_count, int& max_node_id) {
    ifstream fin(filename);
    int from, to;
    max_node_id = 0;
    all_nodes.clear();
    while (fin >> from >> to) {
        max_node_id = max(max_node_id, max(from, to));
        all_nodes.insert(from);
        all_nodes.insert(to);
    }
    fin.close();
    int step = (max_node_id + block_num - 1) / block_num;
    vector<ofstream> block_out(block_num + 1);
    for (int i = 1; i <= block_num; ++i) {
        block_out[i].open("block" + to_string(i));
    }
    fin.open(filename);
    while (fin >> from >> to) {
        int file_num = min((from - 1) / step + 1, block_num);
        block_out[file_num] << from << " " << to << "\n";
    }
    fin.close();
    for (int i = 1; i <= block_num; ++i) block_out[i].close();
    node_count = all_nodes.size();
}

// 统计所有节点的出度，只统计实际节点
void compute_out_degree(int block_num, unordered_map<int, int>& out_degree) {
    for (int node : all_nodes) out_degree[node] = 0;
    for (int i = 1; i <= block_num; ++i) {
        ifstream fin("block" + to_string(i));
        int from, to;
        while (fin >> from >> to) {
            out_degree[from]++;
        }
        fin.close();
    }
}

// PageRank主流程（外存分块，遍历只针对实际节点）
void pagerank_block_external(int block_num, int node_count, int max_node_id, double alpha, vector<pair<int, double>>& result, double tol=EPSILON, int max_iter=MAX_ITER) {
    unordered_map<int, double> pr, pr_next;
    unordered_map<int, int> out_degree;
    for (int node : all_nodes) pr[node] = 1.0 / node_count;
    compute_out_degree(block_num, out_degree);

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int node : all_nodes) pr_next[node] = 0.0;
        double leak = 0.0;
        for (int node : all_nodes) {
            if (out_degree[node] == 0) leak += pr[node];
        }
        for (int blk = 1; blk <= block_num; ++blk) {
            ifstream fin("block" + to_string(blk));
            int from, to;
            while (fin >> from >> to) {
                if (out_degree[from] == 0) continue;
                pr_next[to] += alpha * pr[from] / out_degree[from];
            }
            fin.close();
        }
        double base = (1.0 - alpha) / node_count + alpha * leak / node_count;
        for (int node : all_nodes) {
            pr_next[node] += base;
        }
        double diff = 0.0;
        for (int node : all_nodes) diff += fabs(pr_next[node] - pr[node]);
        pr.swap(pr_next);
        if (diff < tol) break;
    }
    result.clear();
    for (auto& kv : pr) result.push_back({kv.first, kv.second});
    sort(result.begin(), result.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return a.second > b.second;
    });
    if (result.size() > 100) result.resize(100);
}

int main() {
    int node_count = 0, max_node_id = 0;
    preprocess_blocks("Data.txt", BLOCK_NUM, node_count, max_node_id);
    vector<pair<int, double>> result;
    pagerank_block_external(BLOCK_NUM, node_count, max_node_id, TELEPORT_ALPHA, result);
    // 输出结果
    ofstream fout("Res.txt", ios::out | ios::trunc);
    fout << fixed << setprecision(8);
    for (const auto& kv : result) {
        fout << kv.first << " " << kv.second << endl;
    }
    fout.close();
    // 删除所有block分块文件
    for (int i = 1; i <= BLOCK_NUM; ++i) {
        string fname = "block" + to_string(i);
        remove(fname.c_str());
    }
    cout << "PageRank finished, written to Res.txt" << endl;
    return 0;
}

// NOTE: 当前为真正的外存分块实现，单次内存仅需存储一个块的边和全部节点的PR值，极大降低内存峰值。
// 若需进一步优化，可将PR数组也分块存储。