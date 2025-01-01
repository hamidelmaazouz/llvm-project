#include "Q1ISelDAGToDAG.h"
#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "Q1.h"
#include "Q1TargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "Q1-isel"
#define PASS_NAME "Q1 DAG->DAG Pattern Instruction Selection"

FunctionPass *llvm::createQ1ISelDag(Q1TargetMachine &TM,
                                    CodeGenOptLevel OptLevel) {
  return new Q1DAGToDAGISelLegacy(TM, OptLevel);
}

void Q1DAGToDAGISel::Select(SDNode *Node) {
  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.

  // Select the default instruction.
  SelectCode(Node);
}

char Q1DAGToDAGISelLegacy::ID = 0;

Q1DAGToDAGISelLegacy::Q1DAGToDAGISelLegacy(Q1TargetMachine &TM,
                                           CodeGenOptLevel OptLevel)
    : SelectionDAGISelLegacy(ID,
                             std::make_unique<Q1DAGToDAGISel>(TM, OptLevel)) {}

INITIALIZE_PASS(Q1DAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)
