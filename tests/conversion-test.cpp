#include <iostream>
#include <ROOT>

void fill_tree(const char *treeName, const char *fileName) {
   ROOT::RDataFrame d(10);
   d.Define("b1", [](ULong64_t entry) -> double { return entry; }, {"rdfentry_"})
      .Define("b2", [](ULong64_t entry) -> int { return entry * entry; }, {"rdfentry_"})
      .Snapshot(treeName, fileName);
}

int main() {
    std::cout << "Hello, World\n";
    return 0;
}
