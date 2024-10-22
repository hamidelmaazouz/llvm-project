#include "Q1ISelDAGToDAG.h"
#include "MCTargetDesc/Q1BaseInfo.h"
#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "MCTargetDesc/Q1MatInt.h"
#include "Q1ISelLowering.h"
#include "Q1InstrInfo.h"
#include "Q1MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/IntrinsicsQ1.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "Q1-isel"
#define PASS_NAME "RISC-V DAG->DAG Pattern Instruction Selection"

static cl::opt<bool> UsePseudoMovImm(
    "Q1-use-rematerializable-movimm", cl::Hidden,
    cl::desc("Use a rematerializable pseudoinstruction for 2 instruction "
             "constant materialization"),
    cl::init(false));

namespace llvm::Q1 {
#define GET_Q1VSSEGTable_IMPL
#define GET_Q1VLSEGTable_IMPL
#define GET_Q1VLXSEGTable_IMPL
#define GET_Q1VSXSEGTable_IMPL
#define GET_Q1VLETable_IMPL
#define GET_Q1VSETable_IMPL
#define GET_Q1VLXTable_IMPL
#define GET_Q1VSXTable_IMPL
#include "Q1GenSearchableTables.inc"
} // namespace llvm::Q1

void Q1DAGToDAGISel::PreprocessISelDAG() {
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty())
      continue;

    SDValue Result;
    switch (N->getOpcode()) {
    case ISD::SPLAT_VECTOR: {
      // Convert integer SPLAT_VECTOR to VMV_V_X_VL and floating-point
      // SPLAT_VECTOR to VFMV_V_F_VL to reduce isel burden.
      MVT VT = N->getSimpleValueType(0);
      unsigned Opc =
          VT.isInteger() ? Q1ISD::VMV_V_X_VL : Q1ISD::VFMV_V_F_VL;
      SDLoc DL(N);
      SDValue VL = CurDAG->getRegister(Q1::X0, Subtarget->getXLenVT());
      SDValue Src = N->getOperand(0);
      if (VT.isInteger())
        Src = CurDAG->getNode(ISD::ANY_EXTEND, DL, Subtarget->getXLenVT(),
                              N->getOperand(0));
      Result = CurDAG->getNode(Opc, DL, VT, CurDAG->getUNDEF(VT), Src, VL);
      break;
    }
    case Q1ISD::SPLAT_VECTOR_SPLIT_I64_VL: {
      // Lower SPLAT_VECTOR_SPLIT_I64 to two scalar stores and a stride 0 vector
      // load. Done after lowering and combining so that we have a chance to
      // optimize this to VMV_V_X_VL when the upper bits aren't needed.
      assert(N->getNumOperands() == 4 && "Unexpected number of operands");
      MVT VT = N->getSimpleValueType(0);
      SDValue Passthru = N->getOperand(0);
      SDValue Lo = N->getOperand(1);
      SDValue Hi = N->getOperand(2);
      SDValue VL = N->getOperand(3);
      assert(VT.getVectorElementType() == MVT::i64 && VT.isScalableVector() &&
             Lo.getValueType() == MVT::i32 && Hi.getValueType() == MVT::i32 &&
             "Unexpected VTs!");
      MachineFunction &MF = CurDAG->getMachineFunction();
      SDLoc DL(N);

      // Create temporary stack for each expanding node.
      SDValue StackSlot =
          CurDAG->CreateStackTemporary(TypeSize::getFixed(8), Align(8));
      int FI = cast<FrameIndexSDNode>(StackSlot.getNode())->getIndex();
      MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);

      SDValue Chain = CurDAG->getEntryNode();
      Lo = CurDAG->getStore(Chain, DL, Lo, StackSlot, MPI, Align(8));

      SDValue OffsetSlot =
          CurDAG->getMemBasePlusOffset(StackSlot, TypeSize::getFixed(4), DL);
      Hi = CurDAG->getStore(Chain, DL, Hi, OffsetSlot, MPI.getWithOffset(4),
                            Align(8));

      Chain = CurDAG->getNode(ISD::TokenFactor, DL, MVT::Other, Lo, Hi);

      SDVTList VTs = CurDAG->getVTList({VT, MVT::Other});
      SDValue IntID =
          CurDAG->getTargetConstant(Intrinsic::Q1_vlse, DL, MVT::i64);
      SDValue Ops[] = {Chain,
                       IntID,
                       Passthru,
                       StackSlot,
                       CurDAG->getRegister(Q1::X0, MVT::i64),
                       VL};

      Result = CurDAG->getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops,
                                           MVT::i64, MPI, Align(8),
                                           MachineMemOperand::MOLoad);
      break;
    }
    }

    if (Result) {
      LLVM_DEBUG(dbgs() << "RISC-V DAG preprocessing replacing:\nOld:    ");
      LLVM_DEBUG(N->dump(CurDAG));
      LLVM_DEBUG(dbgs() << "\nNew: ");
      LLVM_DEBUG(Result->dump(CurDAG));
      LLVM_DEBUG(dbgs() << "\n");

      CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
      MadeChange = true;
    }
  }

  if (MadeChange)
    CurDAG->RemoveDeadNodes();
}

void Q1DAGToDAGISel::PostprocessISelDAG() {
  HandleSDNode Dummy(CurDAG->getRoot());
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    MadeChange |= doPeepholeSExtW(N);

    // FIXME: This is here only because the VMerge transform doesn't
    // know how to handle masked true inputs.  Once that has been moved
    // to post-ISEL, this can be deleted as well.
    MadeChange |= doPeepholeMaskedRVV(cast<MachineSDNode>(N));
  }

  CurDAG->setRoot(Dummy.getValue());

  MadeChange |= doPeepholeMergeVVMFold();

  // After we're done with everything else, convert IMPLICIT_DEF
  // passthru operands to NoRegister.  This is required to workaround
  // an optimization deficiency in MachineCSE.  This really should
  // be merged back into each of the patterns (i.e. there's no good
  // reason not to go directly to NoReg), but is being done this way
  // to allow easy backporting.
  MadeChange |= doPeepholeNoRegPassThru();

  if (MadeChange)
    CurDAG->RemoveDeadNodes();
}

static SDValue selectImmSeq(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                            Q1MatInt::InstSeq &Seq) {
  SDValue SrcReg = CurDAG->getRegister(Q1::X0, VT);
  for (const Q1MatInt::Inst &Inst : Seq) {
    SDValue SDImm =
        CurDAG->getSignedConstant(Inst.getImm(), DL, VT, /*isTarget=*/true);
    SDNode *Result = nullptr;
    switch (Inst.getOpndKind()) {
    case Q1MatInt::Imm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SDImm);
      break;
    case Q1MatInt::RegX0:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg,
                                      CurDAG->getRegister(Q1::X0, VT));
      break;
    case Q1MatInt::RegReg:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SrcReg);
      break;
    case Q1MatInt::RegImm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SDImm);
      break;
    }

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return SrcReg;
}

static SDValue selectImm(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                         int64_t Imm, const Q1Subtarget &Subtarget) {
  Q1MatInt::InstSeq Seq = Q1MatInt::generateInstSeq(Imm, Subtarget);

  // Use a rematerializable pseudo instruction for short sequences if enabled.
  if (Seq.size() == 2 && UsePseudoMovImm)
    return SDValue(CurDAG->getMachineNode(Q1::PseudoMovImm, DL, VT,
                                          CurDAG->getSignedConstant(
                                              Imm, DL, VT, /*isTarget=*/true)),
                   0);

  // See if we can create this constant as (ADD (SLLI X, C), X) where X is at
  // worst an LUI+ADDIW. This will require an extra register, but avoids a
  // constant pool.
  // If we have Zba we can use (ADD_UW X, (SLLI X, 32)) to handle cases where
  // low and high 32 bits are the same and bit 31 and 63 are set.
  if (Seq.size() > 3) {
    unsigned ShiftAmt, AddOpc;
    Q1MatInt::InstSeq SeqLo =
        Q1MatInt::generateTwoRegInstSeq(Imm, Subtarget, ShiftAmt, AddOpc);
    if (!SeqLo.empty() && (SeqLo.size() + 2) < Seq.size()) {
      SDValue Lo = selectImmSeq(CurDAG, DL, VT, SeqLo);

      SDValue SLLI = SDValue(
          CurDAG->getMachineNode(Q1::SLLI, DL, VT, Lo,
                                 CurDAG->getTargetConstant(ShiftAmt, DL, VT)),
          0);
      return SDValue(CurDAG->getMachineNode(AddOpc, DL, VT, Lo, SLLI), 0);
    }
  }

  // Otherwise, use the original sequence.
  return selectImmSeq(CurDAG, DL, VT, Seq);
}

void Q1DAGToDAGISel::addVectorLoadStoreOperands(
    SDNode *Node, unsigned Log2SEW, const SDLoc &DL, unsigned CurOp,
    bool IsMasked, bool IsStridedOrIndexed, SmallVectorImpl<SDValue> &Operands,
    bool IsLoad, MVT *IndexVT) {
  SDValue Chain = Node->getOperand(0);
  SDValue Glue;

  Operands.push_back(Node->getOperand(CurOp++)); // Base pointer.

  if (IsStridedOrIndexed) {
    Operands.push_back(Node->getOperand(CurOp++)); // Index.
    if (IndexVT)
      *IndexVT = Operands.back()->getSimpleValueType(0);
  }

  if (IsMasked) {
    // Mask needs to be copied to V0.
    SDValue Mask = Node->getOperand(CurOp++);
    Chain = CurDAG->getCopyToReg(Chain, DL, Q1::V0, Mask, SDValue());
    Glue = Chain.getValue(1);
    Operands.push_back(CurDAG->getRegister(Q1::V0, Mask.getValueType()));
  }
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);

  MVT XLenVT = Subtarget->getXLenVT();
  SDValue SEWOp = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);
  Operands.push_back(SEWOp);

  // At the IR layer, all the masked load intrinsics have policy operands,
  // none of the others do.  All have passthru operands.  For our pseudos,
  // all loads have policy operands.
  if (IsLoad) {
    uint64_t Policy = Q1II::MASK_AGNOSTIC;
    if (IsMasked)
      Policy = Node->getConstantOperandVal(CurOp++);
    SDValue PolicyOp = CurDAG->getTargetConstant(Policy, DL, XLenVT);
    Operands.push_back(PolicyOp);
  }

  Operands.push_back(Chain); // Chain.
  if (Glue)
    Operands.push_back(Glue);
}

void Q1DAGToDAGISel::selectVLSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands, /*IsLoad=*/true);

  const Q1::VLSEGPseudo *P =
      Q1::getVLSEGPseudo(NF, IsMasked, IsStrided, /*FF*/ false, Log2SEW,
                            static_cast<unsigned>(LMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  ReplaceUses(SDValue(Node, 0), SDValue(Load, 0));
  ReplaceUses(SDValue(Node, 1), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void Q1DAGToDAGISel::selectVLSEGFF(SDNode *Node, unsigned NF,
                                      bool IsMasked) {
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  MVT XLenVT = Subtarget->getXLenVT();
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 7> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ false, Operands,
                             /*IsLoad=*/true);

  const Q1::VLSEGPseudo *P =
      Q1::getVLSEGPseudo(NF, IsMasked, /*Strided*/ false, /*FF*/ true,
                            Log2SEW, static_cast<unsigned>(LMUL));
  MachineSDNode *Load = CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped,
                                               XLenVT, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  ReplaceUses(SDValue(Node, 0), SDValue(Load, 0)); // Result
  ReplaceUses(SDValue(Node, 1), SDValue(Load, 1)); // VL
  ReplaceUses(SDValue(Node, 2), SDValue(Load, 2)); // Chain
  CurDAG->RemoveDeadNode(Node);
}

void Q1DAGToDAGISel::selectVLXSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands,
                             /*IsLoad=*/true, &IndexVT);

#ifndef NDEBUG
  // Number of element = RVVBitsPerBlock * LMUL / SEW
  unsigned ContainedTyNumElts = Q1::RVVBitsPerBlock >> Log2SEW;
  auto DecodedLMUL = Q1VType::decodeVLMUL(LMUL);
  if (DecodedLMUL.second)
    ContainedTyNumElts /= DecodedLMUL.first;
  else
    ContainedTyNumElts *= DecodedLMUL.first;
  assert(ContainedTyNumElts == IndexVT.getVectorMinNumElements() &&
         "Element count mismatch");
#endif

  Q1II::VLMUL IndexLMUL = Q1TargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
    report_fatal_error("The V extension does not support EEW=64 for index "
                       "values when XLEN=32");
  }
  const Q1::VLXSEGPseudo *P = Q1::getVLXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

  ReplaceUses(SDValue(Node, 0), SDValue(Load, 0));
  ReplaceUses(SDValue(Node, 1), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void Q1DAGToDAGISel::selectVSSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands);

  const Q1::VSSEGPseudo *P = Q1::getVSSEGPseudo(
      NF, IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

  ReplaceNode(Node, Store);
}

void Q1DAGToDAGISel::selectVSXSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands,
                             /*IsLoad=*/false, &IndexVT);

#ifndef NDEBUG
  // Number of element = RVVBitsPerBlock * LMUL / SEW
  unsigned ContainedTyNumElts = Q1::RVVBitsPerBlock >> Log2SEW;
  auto DecodedLMUL = Q1VType::decodeVLMUL(LMUL);
  if (DecodedLMUL.second)
    ContainedTyNumElts /= DecodedLMUL.first;
  else
    ContainedTyNumElts *= DecodedLMUL.first;
  assert(ContainedTyNumElts == IndexVT.getVectorMinNumElements() &&
         "Element count mismatch");
#endif

  Q1II::VLMUL IndexLMUL = Q1TargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
    report_fatal_error("The V extension does not support EEW=64 for index "
                       "values when XLEN=32");
  }
  const Q1::VSXSEGPseudo *P = Q1::getVSXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  if (auto *MemOp = dyn_cast<MemSDNode>(Node))
    CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

  ReplaceNode(Node, Store);
}

void Q1DAGToDAGISel::selectVSETVLI(SDNode *Node) {
  if (!Subtarget->hasVInstructions())
    return;

  assert(Node->getOpcode() == ISD::INTRINSIC_WO_CHAIN && "Unexpected opcode");

  SDLoc DL(Node);
  MVT XLenVT = Subtarget->getXLenVT();

  unsigned IntNo = Node->getConstantOperandVal(0);

  assert((IntNo == Intrinsic::Q1_vsetvli ||
          IntNo == Intrinsic::Q1_vsetvlimax) &&
         "Unexpected vsetvli intrinsic");

  bool VLMax = IntNo == Intrinsic::Q1_vsetvlimax;
  unsigned Offset = (VLMax ? 1 : 2);

  assert(Node->getNumOperands() == Offset + 2 &&
         "Unexpected number of operands");

  unsigned SEW =
      Q1VType::decodeVSEW(Node->getConstantOperandVal(Offset) & 0x7);
  Q1II::VLMUL VLMul = static_cast<Q1II::VLMUL>(
      Node->getConstantOperandVal(Offset + 1) & 0x7);

  unsigned VTypeI = Q1VType::encodeVTYPE(VLMul, SEW, /*TailAgnostic*/ true,
                                            /*MaskAgnostic*/ true);
  SDValue VTypeIOp = CurDAG->getTargetConstant(VTypeI, DL, XLenVT);

  SDValue VLOperand;
  unsigned Opcode = Q1::PseudoVSETVLI;
  if (auto *C = dyn_cast<ConstantSDNode>(Node->getOperand(1))) {
    if (auto VLEN = Subtarget->getRealVLen())
      if (*VLEN / Q1VType::getSEWLMULRatio(SEW, VLMul) == C->getZExtValue())
        VLMax = true;
  }
  if (VLMax || isAllOnesConstant(Node->getOperand(1))) {
    VLOperand = CurDAG->getRegister(Q1::X0, XLenVT);
    Opcode = Q1::PseudoVSETVLIX0;
  } else {
    VLOperand = Node->getOperand(1);

    if (auto *C = dyn_cast<ConstantSDNode>(VLOperand)) {
      uint64_t AVL = C->getZExtValue();
      if (isUInt<5>(AVL)) {
        SDValue VLImm = CurDAG->getTargetConstant(AVL, DL, XLenVT);
        ReplaceNode(Node, CurDAG->getMachineNode(Q1::PseudoVSETIVLI, DL,
                                                 XLenVT, VLImm, VTypeIOp));
        return;
      }
    }
  }

  ReplaceNode(Node,
              CurDAG->getMachineNode(Opcode, DL, XLenVT, VLOperand, VTypeIOp));
}

bool Q1DAGToDAGISel::tryShrinkShlLogicImm(SDNode *Node) {
  MVT VT = Node->getSimpleValueType(0);
  unsigned Opcode = Node->getOpcode();
  assert((Opcode == ISD::AND || Opcode == ISD::OR || Opcode == ISD::XOR) &&
         "Unexpected opcode");
  SDLoc DL(Node);

  // For operations of the form (x << C1) op C2, check if we can use
  // ANDI/ORI/XORI by transforming it into (x op (C2>>C1)) << C1.
  SDValue N0 = Node->getOperand(0);
  SDValue N1 = Node->getOperand(1);

  ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(N1);
  if (!Cst)
    return false;

  int64_t Val = Cst->getSExtValue();

  // Check if immediate can already use ANDI/ORI/XORI.
  if (isInt<12>(Val))
    return false;

  SDValue Shift = N0;

  // If Val is simm32 and we have a sext_inreg from i32, then the binop
  // produces at least 33 sign bits. We can peek through the sext_inreg and use
  // a SLLIW at the end.
  bool SignExt = false;
  if (isInt<32>(Val) && N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      N0.hasOneUse() && cast<VTSDNode>(N0.getOperand(1))->getVT() == MVT::i32) {
    SignExt = true;
    Shift = N0.getOperand(0);
  }

  if (Shift.getOpcode() != ISD::SHL || !Shift.hasOneUse())
    return false;

  ConstantSDNode *ShlCst = dyn_cast<ConstantSDNode>(Shift.getOperand(1));
  if (!ShlCst)
    return false;

  uint64_t ShAmt = ShlCst->getZExtValue();

  // Make sure that we don't change the operation by removing bits.
  // This only matters for OR and XOR, AND is unaffected.
  uint64_t RemovedBitsMask = maskTrailingOnes<uint64_t>(ShAmt);
  if (Opcode != ISD::AND && (Val & RemovedBitsMask) != 0)
    return false;

  int64_t ShiftedVal = Val >> ShAmt;
  if (!isInt<12>(ShiftedVal))
    return false;

  // If we peeked through a sext_inreg, make sure the shift is valid for SLLIW.
  if (SignExt && ShAmt >= 32)
    return false;

  // Ok, we can reorder to get a smaller immediate.
  unsigned BinOpc;
  switch (Opcode) {
  default: llvm_unreachable("Unexpected opcode");
  case ISD::AND: BinOpc = Q1::ANDI; break;
  case ISD::OR:  BinOpc = Q1::ORI;  break;
  case ISD::XOR: BinOpc = Q1::XORI; break;
  }

  unsigned ShOpc = SignExt ? Q1::SLLIW : Q1::SLLI;

  SDNode *BinOp = CurDAG->getMachineNode(
      BinOpc, DL, VT, Shift.getOperand(0),
      CurDAG->getSignedConstant(ShiftedVal, DL, VT, /*isTarget=*/true));
  SDNode *SLLI =
      CurDAG->getMachineNode(ShOpc, DL, VT, SDValue(BinOp, 0),
                             CurDAG->getTargetConstant(ShAmt, DL, VT));
  ReplaceNode(Node, SLLI);
  return true;
}

bool Q1DAGToDAGISel::trySignedBitfieldExtract(SDNode *Node) {
  // Only supported with XTHeadBb at the moment.
  if (!Subtarget->hasVendorXTHeadBb())
    return false;

  auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
  if (!N1C)
    return false;

  SDValue N0 = Node->getOperand(0);
  if (!N0.hasOneUse())
    return false;

  auto BitfieldExtract = [&](SDValue N0, unsigned Msb, unsigned Lsb, SDLoc DL,
                             MVT VT) {
    return CurDAG->getMachineNode(Q1::TH_EXT, DL, VT, N0.getOperand(0),
                                  CurDAG->getTargetConstant(Msb, DL, VT),
                                  CurDAG->getTargetConstant(Lsb, DL, VT));
  };

  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  const unsigned RightShAmt = N1C->getZExtValue();

  // Transform (sra (shl X, C1) C2) with C1 < C2
  //        -> (TH.EXT X, msb, lsb)
  if (N0.getOpcode() == ISD::SHL) {
    auto *N01C = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    if (!N01C)
      return false;

    const unsigned LeftShAmt = N01C->getZExtValue();
    // Make sure that this is a bitfield extraction (i.e., the shift-right
    // amount can not be less than the left-shift).
    if (LeftShAmt > RightShAmt)
      return false;

    const unsigned MsbPlusOne = VT.getSizeInBits() - LeftShAmt;
    const unsigned Msb = MsbPlusOne - 1;
    const unsigned Lsb = RightShAmt - LeftShAmt;

    SDNode *TH_EXT = BitfieldExtract(N0, Msb, Lsb, DL, VT);
    ReplaceNode(Node, TH_EXT);
    return true;
  }

  // Transform (sra (sext_inreg X, _), C) ->
  //           (TH.EXT X, msb, lsb)
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    unsigned ExtSize =
        cast<VTSDNode>(N0.getOperand(1))->getVT().getSizeInBits();

    // ExtSize of 32 should use sraiw via tablegen pattern.
    if (ExtSize == 32)
      return false;

    const unsigned Msb = ExtSize - 1;
    const unsigned Lsb = RightShAmt;

    SDNode *TH_EXT = BitfieldExtract(N0, Msb, Lsb, DL, VT);
    ReplaceNode(Node, TH_EXT);
    return true;
  }

  return false;
}

bool Q1DAGToDAGISel::tryIndexedLoad(SDNode *Node) {
  // Target does not support indexed loads.
  if (!Subtarget->hasVendorXTHeadMemIdx())
    return false;

  LoadSDNode *Ld = cast<LoadSDNode>(Node);
  ISD::MemIndexedMode AM = Ld->getAddressingMode();
  if (AM == ISD::UNINDEXED)
    return false;

  const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Ld->getOffset());
  if (!C)
    return false;

  EVT LoadVT = Ld->getMemoryVT();
  assert((AM == ISD::PRE_INC || AM == ISD::POST_INC) &&
         "Unexpected addressing mode");
  bool IsPre = AM == ISD::PRE_INC;
  bool IsPost = AM == ISD::POST_INC;
  int64_t Offset = C->getSExtValue();

  // The constants that can be encoded in the THeadMemIdx instructions
  // are of the form (sign_extend(imm5) << imm2).
  int64_t Shift;
  for (Shift = 0; Shift < 4; Shift++)
    if (isInt<5>(Offset >> Shift) && ((Offset % (1LL << Shift)) == 0))
      break;

  // Constant cannot be encoded.
  if (Shift == 4)
    return false;

  bool IsZExt = (Ld->getExtensionType() == ISD::ZEXTLOAD);
  unsigned Opcode;
  if (LoadVT == MVT::i8 && IsPre)
    Opcode = IsZExt ? Q1::TH_LBUIB : Q1::TH_LBIB;
  else if (LoadVT == MVT::i8 && IsPost)
    Opcode = IsZExt ? Q1::TH_LBUIA : Q1::TH_LBIA;
  else if (LoadVT == MVT::i16 && IsPre)
    Opcode = IsZExt ? Q1::TH_LHUIB : Q1::TH_LHIB;
  else if (LoadVT == MVT::i16 && IsPost)
    Opcode = IsZExt ? Q1::TH_LHUIA : Q1::TH_LHIA;
  else if (LoadVT == MVT::i32 && IsPre)
    Opcode = IsZExt ? Q1::TH_LWUIB : Q1::TH_LWIB;
  else if (LoadVT == MVT::i32 && IsPost)
    Opcode = IsZExt ? Q1::TH_LWUIA : Q1::TH_LWIA;
  else if (LoadVT == MVT::i64 && IsPre)
    Opcode = Q1::TH_LDIB;
  else if (LoadVT == MVT::i64 && IsPost)
    Opcode = Q1::TH_LDIA;
  else
    return false;

  EVT Ty = Ld->getOffset().getValueType();
  SDValue Ops[] = {Ld->getBasePtr(),
                   CurDAG->getSignedConstant(Offset >> Shift, SDLoc(Node), Ty,
                                             /*isTarget=*/true),
                   CurDAG->getTargetConstant(Shift, SDLoc(Node), Ty),
                   Ld->getChain()};
  SDNode *New = CurDAG->getMachineNode(Opcode, SDLoc(Node), Ld->getValueType(0),
                                       Ld->getValueType(1), MVT::Other, Ops);

  MachineMemOperand *MemOp = cast<MemSDNode>(Node)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(New), {MemOp});

  ReplaceNode(Node, New);

  return true;
}

void Q1DAGToDAGISel::selectSF_VC_X_SE(SDNode *Node) {
  if (!Subtarget->hasVInstructions())
    return;

  assert(Node->getOpcode() == ISD::INTRINSIC_VOID && "Unexpected opcode");

  SDLoc DL(Node);
  unsigned IntNo = Node->getConstantOperandVal(1);

  assert((IntNo == Intrinsic::Q1_sf_vc_x_se ||
          IntNo == Intrinsic::Q1_sf_vc_i_se) &&
         "Unexpected vsetvli intrinsic");

  // imm, imm, imm, simm5/scalar, sew, log2lmul, vl
  unsigned Log2SEW = Log2_32(Node->getConstantOperandVal(6));
  SDValue SEWOp =
      CurDAG->getTargetConstant(Log2SEW, DL, Subtarget->getXLenVT());
  SmallVector<SDValue, 8> Operands = {Node->getOperand(2), Node->getOperand(3),
                                      Node->getOperand(4), Node->getOperand(5),
                                      Node->getOperand(8), SEWOp,
                                      Node->getOperand(0)};

  unsigned Opcode;
  auto *LMulSDNode = cast<ConstantSDNode>(Node->getOperand(7));
  switch (LMulSDNode->getSExtValue()) {
  case 5:
    Opcode = IntNo == Intrinsic::Q1_sf_vc_x_se ? Q1::PseudoVC_X_SE_MF8
                                                  : Q1::PseudoVC_I_SE_MF8;
    break;
  case 6:
    Opcode = IntNo == Intrinsic::Q1_sf_vc_x_se ? Q1::PseudoVC_X_SE_MF4
                                                  : Q1::PseudoVC_I_SE_MF4;
    break;
  case 7:
    Opcode = IntNo == Intrinsic::Q1_sf_vc_x_se ? Q1::PseudoVC_X_SE_MF2
                                                  : Q1::PseudoVC_I_SE_MF2;
    break;
  case 0:
    Opcode = IntNo == Intrinsic::Q1_sf_vc_x_se ? Q1::PseudoVC_X_SE_M1
                                                  : Q1::PseudoVC_I_SE_M1;
    break;
  case 1:
    Opcode = IntNo == Intrinsic::Q1_sf_vc_x_se ? Q1::PseudoVC_X_SE_M2
                                                  : Q1::PseudoVC_I_SE_M2;
    break;
  case 2:
    Opcode = IntNo == Intrinsic::Q1_sf_vc_x_se ? Q1::PseudoVC_X_SE_M4
                                                  : Q1::PseudoVC_I_SE_M4;
    break;
  case 3:
    Opcode = IntNo == Intrinsic::Q1_sf_vc_x_se ? Q1::PseudoVC_X_SE_M8
                                                  : Q1::PseudoVC_I_SE_M8;
    break;
  }

  ReplaceNode(Node, CurDAG->getMachineNode(
                        Opcode, DL, Node->getSimpleValueType(0), Operands));
}

static unsigned getSegInstNF(unsigned Intrinsic) {
#define INST_NF_CASE(NAME, NF)                                                 \
  case Intrinsic::Q1_##NAME##NF:                                            \
    return NF;
#define INST_NF_CASE_MASK(NAME, NF)                                            \
  case Intrinsic::Q1_##NAME##NF##_mask:                                     \
    return NF;
#define INST_NF_CASE_FF(NAME, NF)                                              \
  case Intrinsic::Q1_##NAME##NF##ff:                                        \
    return NF;
#define INST_NF_CASE_FF_MASK(NAME, NF)                                         \
  case Intrinsic::Q1_##NAME##NF##ff_mask:                                   \
    return NF;
#define INST_ALL_NF_CASE_BASE(MACRO_NAME, NAME)                                \
  MACRO_NAME(NAME, 2)                                                          \
  MACRO_NAME(NAME, 3)                                                          \
  MACRO_NAME(NAME, 4)                                                          \
  MACRO_NAME(NAME, 5)                                                          \
  MACRO_NAME(NAME, 6)                                                          \
  MACRO_NAME(NAME, 7)                                                          \
  MACRO_NAME(NAME, 8)
#define INST_ALL_NF_CASE(NAME)                                                 \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE, NAME)                                    \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE_MASK, NAME)
#define INST_ALL_NF_CASE_WITH_FF(NAME)                                         \
  INST_ALL_NF_CASE(NAME)                                                       \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE_FF, NAME)                                 \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE_FF_MASK, NAME)
  switch (Intrinsic) {
  default:
    llvm_unreachable("Unexpected segment load/store intrinsic");
    INST_ALL_NF_CASE_WITH_FF(vlseg)
    INST_ALL_NF_CASE(vlsseg)
    INST_ALL_NF_CASE(vloxseg)
    INST_ALL_NF_CASE(vluxseg)
    INST_ALL_NF_CASE(vsseg)
    INST_ALL_NF_CASE(vssseg)
    INST_ALL_NF_CASE(vsoxseg)
    INST_ALL_NF_CASE(vsuxseg)
  }
}

void Q1DAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();
  MVT XLenVT = Subtarget->getXLenVT();
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);

  bool HasBitTest = Subtarget->hasStdExtZbs() || Subtarget->hasVendorXTHeadBs();

  switch (Opcode) {
  case ISD::Constant: {
    assert((VT == Subtarget->getXLenVT() || VT == MVT::i32) && "Unexpected VT");
    auto *ConstNode = cast<ConstantSDNode>(Node);
    if (ConstNode->isZero()) {
      SDValue New =
          CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL, Q1::X0, VT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    // If only the lower 8 bits are used, try to convert this to a simm6 by
    // sign-extending bit 7. This is neutral without the C extension, and
    // allows C.LI to be used if C is present.
    if (isUInt<8>(Imm) && isInt<6>(SignExtend64<8>(Imm)) && hasAllBUsers(Node))
      Imm = SignExtend64<8>(Imm);
    // If the upper XLen-16 bits are not used, try to convert this to a simm12
    // by sign extending bit 15.
    if (isUInt<16>(Imm) && isInt<12>(SignExtend64<16>(Imm)) &&
        hasAllHUsers(Node))
      Imm = SignExtend64<16>(Imm);
    // If the upper 32-bits are not used try to convert this into a simm32 by
    // sign extending bit 32.
    if (!isInt<32>(Imm) && isUInt<32>(Imm) && hasAllWUsers(Node))
      Imm = SignExtend64<32>(Imm);

    ReplaceNode(Node, selectImm(CurDAG, DL, VT, Imm, *Subtarget).getNode());
    return;
  }
  case ISD::ConstantFP: {
    const APFloat &APF = cast<ConstantFPSDNode>(Node)->getValueAPF();

    bool NegZeroF64 = APF.isNegZero() && VT == MVT::f64;
    SDValue Imm;
    // For +0.0 or f64 -0.0 we need to start from X0. For all others, we will
    // create an integer immediate.
    if (APF.isPosZero() || NegZeroF64)
      Imm = CurDAG->getRegister(Q1::X0, XLenVT);
    else
      Imm = selectImm(CurDAG, DL, XLenVT, APF.bitcastToAPInt().getSExtValue(),
                      *Subtarget);

    bool HasZdinx = Subtarget->hasStdExtZdinx();
    bool Is64Bit = Subtarget->is64Bit();
    unsigned Opc;
    switch (VT.SimpleTy) {
    default:
      llvm_unreachable("Unexpected size");
    case MVT::bf16:
      assert(Subtarget->hasStdExtZfbfmin());
      Opc = Q1::FMV_H_X;
      break;
    case MVT::f16:
      Opc = Subtarget->hasStdExtZhinxmin() ? Q1::COPY : Q1::FMV_H_X;
      break;
    case MVT::f32:
      Opc = Subtarget->hasStdExtZfinx() ? Q1::COPY : Q1::FMV_W_X;
      break;
    case MVT::f64:
      // For RV32, we can't move from a GPR, we need to convert instead. This
      // should only happen for +0.0 and -0.0.
      assert((Subtarget->is64Bit() || APF.isZero()) && "Unexpected constant");
      if (Is64Bit)
        Opc = HasZdinx ? Q1::COPY : Q1::FMV_D_X;
      else
        Opc = HasZdinx ? Q1::FCVT_D_W_IN32X : Q1::FCVT_D_W;
      break;
    }

    SDNode *Res;
    if (VT.SimpleTy == MVT::f16 && Opc == Q1::COPY) {
      Res =
          CurDAG->getTargetExtractSubreg(Q1::sub_16, DL, VT, Imm).getNode();
    } else if (VT.SimpleTy == MVT::f32 && Opc == Q1::COPY) {
      Res =
          CurDAG->getTargetExtractSubreg(Q1::sub_32, DL, VT, Imm).getNode();
    } else if (Opc == Q1::FCVT_D_W_IN32X || Opc == Q1::FCVT_D_W)
      Res = CurDAG->getMachineNode(
          Opc, DL, VT, Imm,
          CurDAG->getTargetConstant(Q1FPRndMode::RNE, DL, XLenVT));
    else
      Res = CurDAG->getMachineNode(Opc, DL, VT, Imm);

    // For f64 -0.0, we need to insert a fneg.d idiom.
    if (NegZeroF64) {
      Opc = Q1::FSGNJN_D;
      if (HasZdinx)
        Opc = Is64Bit ? Q1::FSGNJN_D_INX : Q1::FSGNJN_D_IN32X;
      Res =
          CurDAG->getMachineNode(Opc, DL, VT, SDValue(Res, 0), SDValue(Res, 0));
    }

    ReplaceNode(Node, Res);
    return;
  }
  case Q1ISD::BuildPairF64: {
    if (!Subtarget->hasStdExtZdinx())
      break;

    assert(!Subtarget->is64Bit() && "Unexpected subtarget");

    SDValue Ops[] = {
        CurDAG->getTargetConstant(Q1::GPRPairRegClassID, DL, MVT::i32),
        Node->getOperand(0),
        CurDAG->getTargetConstant(Q1::sub_gpr_even, DL, MVT::i32),
        Node->getOperand(1),
        CurDAG->getTargetConstant(Q1::sub_gpr_odd, DL, MVT::i32)};

    SDNode *N =
        CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::f64, Ops);
    ReplaceNode(Node, N);
    return;
  }
  case Q1ISD::SplitF64: {
    if (Subtarget->hasStdExtZdinx()) {
      assert(!Subtarget->is64Bit() && "Unexpected subtarget");

      if (!SDValue(Node, 0).use_empty()) {
        SDValue Lo = CurDAG->getTargetExtractSubreg(Q1::sub_gpr_even, DL, VT,
                                                    Node->getOperand(0));
        ReplaceUses(SDValue(Node, 0), Lo);
      }

      if (!SDValue(Node, 1).use_empty()) {
        SDValue Hi = CurDAG->getTargetExtractSubreg(Q1::sub_gpr_odd, DL, VT,
                                                    Node->getOperand(0));
        ReplaceUses(SDValue(Node, 1), Hi);
      }

      CurDAG->RemoveDeadNode(Node);
      return;
    }

    if (!Subtarget->hasStdExtZfa())
      break;
    assert(Subtarget->hasStdExtD() && !Subtarget->is64Bit() &&
           "Unexpected subtarget");

    // With Zfa, lower to fmv.x.w and fmvh.x.d.
    if (!SDValue(Node, 0).use_empty()) {
      SDNode *Lo = CurDAG->getMachineNode(Q1::FMV_X_W_FPR64, DL, VT,
                                          Node->getOperand(0));
      ReplaceUses(SDValue(Node, 0), SDValue(Lo, 0));
    }
    if (!SDValue(Node, 1).use_empty()) {
      SDNode *Hi = CurDAG->getMachineNode(Q1::FMVH_X_D, DL, VT,
                                          Node->getOperand(0));
      ReplaceUses(SDValue(Node, 1), SDValue(Hi, 0));
    }

    CurDAG->RemoveDeadNode(Node);
    return;
  }
  case ISD::SHL: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !N0.hasOneUse() ||
        !isa<ConstantSDNode>(N0.getOperand(1)))
      break;
    unsigned ShAmt = N1C->getZExtValue();
    uint64_t Mask = N0.getConstantOperandVal(1);

    // Optimize (shl (and X, C2), C) -> (slli (srliw X, C3), C3+C) where C2 has
    // 32 leading zeros and C3 trailing zeros.
    if (ShAmt <= 32 && isShiftedMask_64(Mask)) {
      unsigned XLen = Subtarget->getXLen();
      unsigned LeadingZeros = XLen - llvm::bit_width(Mask);
      unsigned TrailingZeros = llvm::countr_zero(Mask);
      if (TrailingZeros > 0 && LeadingZeros == 32) {
        SDNode *SRLIW = CurDAG->getMachineNode(
            Q1::SRLIW, DL, VT, N0->getOperand(0),
            CurDAG->getTargetConstant(TrailingZeros, DL, VT));
        SDNode *SLLI = CurDAG->getMachineNode(
            Q1::SLLI, DL, VT, SDValue(SRLIW, 0),
            CurDAG->getTargetConstant(TrailingZeros + ShAmt, DL, VT));
        ReplaceNode(Node, SLLI);
        return;
      }
    }
    break;
  }
  case ISD::SRL: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !isa<ConstantSDNode>(N0.getOperand(1)))
      break;
    unsigned ShAmt = N1C->getZExtValue();
    uint64_t Mask = N0.getConstantOperandVal(1);

    // Optimize (srl (and X, C2), C) -> (slli (srliw X, C3), C3-C) where C2 has
    // 32 leading zeros and C3 trailing zeros.
    if (isShiftedMask_64(Mask) && N0.hasOneUse()) {
      unsigned XLen = Subtarget->getXLen();
      unsigned LeadingZeros = XLen - llvm::bit_width(Mask);
      unsigned TrailingZeros = llvm::countr_zero(Mask);
      if (LeadingZeros == 32 && TrailingZeros > ShAmt) {
        SDNode *SRLIW = CurDAG->getMachineNode(
            Q1::SRLIW, DL, VT, N0->getOperand(0),
            CurDAG->getTargetConstant(TrailingZeros, DL, VT));
        SDNode *SLLI = CurDAG->getMachineNode(
            Q1::SLLI, DL, VT, SDValue(SRLIW, 0),
            CurDAG->getTargetConstant(TrailingZeros - ShAmt, DL, VT));
        ReplaceNode(Node, SLLI);
        return;
      }
    }

    // Optimize (srl (and X, C2), C) ->
    //          (srli (slli X, (XLen-C3), (XLen-C3) + C)
    // Where C2 is a mask with C3 trailing ones.
    // Taking into account that the C2 may have had lower bits unset by
    // SimplifyDemandedBits. This avoids materializing the C2 immediate.
    // This pattern occurs when type legalizing right shifts for types with
    // less than XLen bits.
    Mask |= maskTrailingOnes<uint64_t>(ShAmt);
    if (!isMask_64(Mask))
      break;
    unsigned TrailingOnes = llvm::countr_one(Mask);
    if (ShAmt >= TrailingOnes)
      break;
    // If the mask has 32 trailing ones, use SRLI on RV32 or SRLIW on RV64.
    if (TrailingOnes == 32) {
      SDNode *SRLI = CurDAG->getMachineNode(
          Subtarget->is64Bit() ? Q1::SRLIW : Q1::SRLI, DL, VT,
          N0->getOperand(0), CurDAG->getTargetConstant(ShAmt, DL, VT));
      ReplaceNode(Node, SRLI);
      return;
    }

    // Only do the remaining transforms if the AND has one use.
    if (!N0.hasOneUse())
      break;

    // If C2 is (1 << ShAmt) use bexti or th.tst if possible.
    if (HasBitTest && ShAmt + 1 == TrailingOnes) {
      SDNode *BEXTI = CurDAG->getMachineNode(
          Subtarget->hasStdExtZbs() ? Q1::BEXTI : Q1::TH_TST, DL, VT,
          N0->getOperand(0), CurDAG->getTargetConstant(ShAmt, DL, VT));
      ReplaceNode(Node, BEXTI);
      return;
    }

    unsigned LShAmt = Subtarget->getXLen() - TrailingOnes;
    if (Subtarget->hasVendorXTHeadBb()) {
      SDNode *THEXTU = CurDAG->getMachineNode(
          Q1::TH_EXTU, DL, VT, N0->getOperand(0),
          CurDAG->getTargetConstant(TrailingOnes - 1, DL, VT),
          CurDAG->getTargetConstant(ShAmt, DL, VT));
      ReplaceNode(Node, THEXTU);
      return;
    }

    SDNode *SLLI =
        CurDAG->getMachineNode(Q1::SLLI, DL, VT, N0->getOperand(0),
                               CurDAG->getTargetConstant(LShAmt, DL, VT));
    SDNode *SRLI = CurDAG->getMachineNode(
        Q1::SRLI, DL, VT, SDValue(SLLI, 0),
        CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
    ReplaceNode(Node, SRLI);
    return;
  }
  case ISD::SRA: {
    if (trySignedBitfieldExtract(Node))
      return;

    // Optimize (sra (sext_inreg X, i16), C) ->
    //          (srai (slli X, (XLen-16), (XLen-16) + C)
    // And      (sra (sext_inreg X, i8), C) ->
    //          (srai (slli X, (XLen-8), (XLen-8) + C)
    // This can occur when Zbb is enabled, which makes sext_inreg i16/i8 legal.
    // This transform matches the code we get without Zbb. The shifts are more
    // compressible, and this can help expose CSE opportunities in the sdiv by
    // constant optimization.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::SIGN_EXTEND_INREG || !N0.hasOneUse())
      break;
    unsigned ShAmt = N1C->getZExtValue();
    unsigned ExtSize =
        cast<VTSDNode>(N0.getOperand(1))->getVT().getSizeInBits();
    // ExtSize of 32 should use sraiw via tablegen pattern.
    if (ExtSize >= 32 || ShAmt >= ExtSize)
      break;
    unsigned LShAmt = Subtarget->getXLen() - ExtSize;
    SDNode *SLLI =
        CurDAG->getMachineNode(Q1::SLLI, DL, VT, N0->getOperand(0),
                               CurDAG->getTargetConstant(LShAmt, DL, VT));
    SDNode *SRAI = CurDAG->getMachineNode(
        Q1::SRAI, DL, VT, SDValue(SLLI, 0),
        CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
    ReplaceNode(Node, SRAI);
    return;
  }
  case ISD::OR:
  case ISD::XOR:
    if (tryShrinkShlLogicImm(Node))
      return;

    break;
  case ISD::AND: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;

    SDValue N0 = Node->getOperand(0);

    auto tryUnsignedBitfieldExtract = [&](SDNode *Node, SDLoc DL, MVT VT,
                                          SDValue X, unsigned Msb,
                                          unsigned Lsb) {
      if (!Subtarget->hasVendorXTHeadBb())
        return false;

      SDNode *TH_EXTU = CurDAG->getMachineNode(
          Q1::TH_EXTU, DL, VT, X, CurDAG->getTargetConstant(Msb, DL, VT),
          CurDAG->getTargetConstant(Lsb, DL, VT));
      ReplaceNode(Node, TH_EXTU);
      return true;
    };

    bool LeftShift = N0.getOpcode() == ISD::SHL;
    if (LeftShift || N0.getOpcode() == ISD::SRL) {
      auto *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C)
        break;
      unsigned C2 = C->getZExtValue();
      unsigned XLen = Subtarget->getXLen();
      assert((C2 > 0 && C2 < XLen) && "Unexpected shift amount!");

      // Keep track of whether this is a c.andi. If we can't use c.andi, the
      // shift pair might offer more compression opportunities.
      // TODO: We could check for C extension here, but we don't have many lit
      // tests with the C extension enabled so not checking gets better
      // coverage.
      // TODO: What if ANDI faster than shift?
      bool IsCANDI = isInt<6>(N1C->getSExtValue());

      uint64_t C1 = N1C->getZExtValue();

      // Clear irrelevant bits in the mask.
      if (LeftShift)
        C1 &= maskTrailingZeros<uint64_t>(C2);
      else
        C1 &= maskTrailingOnes<uint64_t>(XLen - C2);

      // Some transforms should only be done if the shift has a single use or
      // the AND would become (srli (slli X, 32), 32)
      bool OneUseOrZExtW = N0.hasOneUse() || C1 == UINT64_C(0xFFFFFFFF);

      SDValue X = N0.getOperand(0);

      // Turn (and (srl x, c2) c1) -> (srli (slli x, c3-c2), c3) if c1 is a mask
      // with c3 leading zeros.
      if (!LeftShift && isMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        if (C2 < Leading) {
          // If the number of leading zeros is C2+32 this can be SRLIW.
          if (C2 + 32 == Leading) {
            SDNode *SRLIW = CurDAG->getMachineNode(
                Q1::SRLIW, DL, VT, X, CurDAG->getTargetConstant(C2, DL, VT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // (and (srl (sexti32 Y), c2), c1) -> (srliw (sraiw Y, 31), c3 - 32)
          // if c1 is a mask with c3 leading zeros and c2 >= 32 and c3-c2==1.
          //
          // This pattern occurs when (i32 (srl (sra 31), c3 - 32)) is type
          // legalized and goes through DAG combine.
          if (C2 >= 32 && (Leading - C2) == 1 && N0.hasOneUse() &&
              X.getOpcode() == ISD::SIGN_EXTEND_INREG &&
              cast<VTSDNode>(X.getOperand(1))->getVT() == MVT::i32) {
            SDNode *SRAIW =
                CurDAG->getMachineNode(Q1::SRAIW, DL, VT, X.getOperand(0),
                                       CurDAG->getTargetConstant(31, DL, VT));
            SDNode *SRLIW = CurDAG->getMachineNode(
                Q1::SRLIW, DL, VT, SDValue(SRAIW, 0),
                CurDAG->getTargetConstant(Leading - 32, DL, VT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // Try to use an unsigned bitfield extract (e.g., th.extu) if
          // available.
          // Transform (and (srl x, C2), C1)
          //        -> (<bfextract> x, msb, lsb)
          //
          // Make sure to keep this below the SRLIW cases, as we always want to
          // prefer the more common instruction.
          const unsigned Msb = llvm::bit_width(C1) + C2 - 1;
          const unsigned Lsb = C2;
          if (tryUnsignedBitfieldExtract(Node, DL, VT, X, Msb, Lsb))
            return;

          // (srli (slli x, c3-c2), c3).
          // Skip if we could use (zext.w (sraiw X, C2)).
          bool Skip = Subtarget->hasStdExtZba() && Leading == 32 &&
                      X.getOpcode() == ISD::SIGN_EXTEND_INREG &&
                      cast<VTSDNode>(X.getOperand(1))->getVT() == MVT::i32;
          // Also Skip if we can use bexti or th.tst.
          Skip |= HasBitTest && Leading == XLen - 1;
          if (OneUseOrZExtW && !Skip) {
            SDNode *SLLI = CurDAG->getMachineNode(
                Q1::SLLI, DL, VT, X,
                CurDAG->getTargetConstant(Leading - C2, DL, VT));
            SDNode *SRLI = CurDAG->getMachineNode(
                Q1::SRLI, DL, VT, SDValue(SLLI, 0),
                CurDAG->getTargetConstant(Leading, DL, VT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      // Turn (and (shl x, c2), c1) -> (srli (slli c2+c3), c3) if c1 is a mask
      // shifted by c2 bits with c3 leading zeros.
      if (LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);

        if (C2 + Leading < XLen &&
            C1 == (maskTrailingOnes<uint64_t>(XLen - (C2 + Leading)) << C2)) {
          // Use slli.uw when possible.
          if ((XLen - (C2 + Leading)) == 32 && Subtarget->hasStdExtZba()) {
            SDNode *SLLI_UW =
                CurDAG->getMachineNode(Q1::SLLI_UW, DL, VT, X,
                                       CurDAG->getTargetConstant(C2, DL, VT));
            ReplaceNode(Node, SLLI_UW);
            return;
          }

          // (srli (slli c2+c3), c3)
          if (OneUseOrZExtW && !IsCANDI) {
            SDNode *SLLI = CurDAG->getMachineNode(
                Q1::SLLI, DL, VT, X,
                CurDAG->getTargetConstant(C2 + Leading, DL, VT));
            SDNode *SRLI = CurDAG->getMachineNode(
                Q1::SRLI, DL, VT, SDValue(SLLI, 0),
                CurDAG->getTargetConstant(Leading, DL, VT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      // Turn (and (shr x, c2), c1) -> (slli (srli x, c2+c3), c3) if c1 is a
      // shifted mask with c2 leading zeros and c3 trailing zeros.
      if (!LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = llvm::countr_zero(C1);
        if (Leading == C2 && C2 + Trailing < XLen && OneUseOrZExtW &&
            !IsCANDI) {
          unsigned SrliOpc = Q1::SRLI;
          // If the input is zexti32 we should use SRLIW.
          if (X.getOpcode() == ISD::AND &&
              isa<ConstantSDNode>(X.getOperand(1)) &&
              X.getConstantOperandVal(1) == UINT64_C(0xFFFFFFFF)) {
            SrliOpc = Q1::SRLIW;
            X = X.getOperand(0);
          }
          SDNode *SRLI = CurDAG->getMachineNode(
              SrliOpc, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Q1::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If the leading zero count is C2+32, we can use SRLIW instead of SRLI.
        if (Leading > 32 && (Leading - 32) == C2 && C2 + Trailing < 32 &&
            OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLIW = CurDAG->getMachineNode(
              Q1::SRLIW, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Q1::SLLI, DL, VT, SDValue(SRLIW, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If we have 32 bits in the mask, we can use SLLI_UW instead of SLLI.
        if (Trailing > 0 && Leading + Trailing == 32 && C2 + Trailing < XLen &&
            OneUseOrZExtW && Subtarget->hasStdExtZba()) {
          SDNode *SRLI = CurDAG->getMachineNode(
              Q1::SRLI, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI_UW = CurDAG->getMachineNode(
              Q1::SLLI_UW, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI_UW);
          return;
        }
      }

      // Turn (and (shl x, c2), c1) -> (slli (srli x, c3-c2), c3) if c1 is a
      // shifted mask with no leading zeros and c3 trailing zeros.
      if (LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = llvm::countr_zero(C1);
        if (Leading == 0 && C2 < Trailing && OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLI = CurDAG->getMachineNode(
              Q1::SRLI, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Q1::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If we have (32-C2) leading zeros, we can use SRLIW instead of SRLI.
        if (C2 < Trailing && Leading + C2 == 32 && OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLIW = CurDAG->getMachineNode(
              Q1::SRLIW, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Q1::SLLI, DL, VT, SDValue(SRLIW, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }

        // If we have 32 bits in the mask, we can use SLLI_UW instead of SLLI.
        if (C2 < Trailing && Leading + Trailing == 32 && OneUseOrZExtW &&
            Subtarget->hasStdExtZba()) {
          SDNode *SRLI = CurDAG->getMachineNode(
              Q1::SRLI, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI_UW = CurDAG->getMachineNode(
              Q1::SLLI_UW, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI_UW);
          return;
        }
      }
    }

    const uint64_t C1 = N1C->getZExtValue();

    if (N0.getOpcode() == ISD::SRA && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.hasOneUse()) {
      unsigned C2 = N0.getConstantOperandVal(1);
      unsigned XLen = Subtarget->getXLen();
      assert((C2 > 0 && C2 < XLen) && "Unexpected shift amount!");

      SDValue X = N0.getOperand(0);

      // Prefer SRAIW + ANDI when possible.
      bool Skip = C2 > 32 && isInt<12>(N1C->getSExtValue()) &&
                  X.getOpcode() == ISD::SHL &&
                  isa<ConstantSDNode>(X.getOperand(1)) &&
                  X.getConstantOperandVal(1) == 32;
      // Turn (and (sra x, c2), c1) -> (srli (srai x, c2-c3), c3) if c1 is a
      // mask with c3 leading zeros and c2 is larger than c3.
      if (isMask_64(C1) && !Skip) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        if (C2 > Leading) {
          SDNode *SRAI = CurDAG->getMachineNode(
              Q1::SRAI, DL, VT, X,
              CurDAG->getTargetConstant(C2 - Leading, DL, VT));
          SDNode *SRLI = CurDAG->getMachineNode(
              Q1::SRLI, DL, VT, SDValue(SRAI, 0),
              CurDAG->getTargetConstant(Leading, DL, VT));
          ReplaceNode(Node, SRLI);
          return;
        }
      }

      // Look for (and (sra y, c2), c1) where c1 is a shifted mask with c3
      // leading zeros and c4 trailing zeros. If c2 is greater than c3, we can
      // use (slli (srli (srai y, c2 - c3), c3 + c4), c4).
      if (isShiftedMask_64(C1) && !Skip) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = llvm::countr_zero(C1);
        if (C2 > Leading && Leading > 0 && Trailing > 0) {
          SDNode *SRAI = CurDAG->getMachineNode(
              Q1::SRAI, DL, VT, N0.getOperand(0),
              CurDAG->getTargetConstant(C2 - Leading, DL, VT));
          SDNode *SRLI = CurDAG->getMachineNode(
              Q1::SRLI, DL, VT, SDValue(SRAI, 0),
              CurDAG->getTargetConstant(Leading + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Q1::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
      }
    }

    // If C1 masks off the upper bits only (but can't be formed as an
    // ANDI), use an unsigned bitfield extract (e.g., th.extu), if
    // available.
    // Transform (and x, C1)
    //        -> (<bfextract> x, msb, lsb)
    if (isMask_64(C1) && !isInt<12>(N1C->getSExtValue())) {
      const unsigned Msb = llvm::bit_width(C1) - 1;
      if (tryUnsignedBitfieldExtract(Node, DL, VT, N0, Msb, 0))
        return;
    }

    if (tryShrinkShlLogicImm(Node))
      return;

    break;
  }
  case ISD::MUL: {
    // Special case for calculating (mul (and X, C2), C1) where the full product
    // fits in XLen bits. We can shift X left by the number of leading zeros in
    // C2 and shift C1 left by XLen-lzcnt(C2). This will ensure the final
    // product has XLen trailing zeros, putting it in the output of MULHU. This
    // can avoid materializing a constant in a register for C2.

    // RHS should be a constant.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C || !N1C->hasOneUse())
      break;

    // LHS should be an AND with constant.
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !isa<ConstantSDNode>(N0.getOperand(1)))
      break;

    uint64_t C2 = N0.getConstantOperandVal(1);

    // Constant should be a mask.
    if (!isMask_64(C2))
      break;

    // If this can be an ANDI or ZEXT.H, don't do this if the ANDI/ZEXT has
    // multiple users or the constant is a simm12. This prevents inserting a
    // shift and still have uses of the AND/ZEXT. Shifting a simm12 will likely
    // make it more costly to materialize. Otherwise, using a SLLI might allow
    // it to be compressed.
    bool IsANDIOrZExt =
        isInt<12>(C2) ||
        (C2 == UINT64_C(0xFFFF) && Subtarget->hasStdExtZbb());
    // With XTHeadBb, we can use TH.EXTU.
    IsANDIOrZExt |= C2 == UINT64_C(0xFFFF) && Subtarget->hasVendorXTHeadBb();
    if (IsANDIOrZExt && (isInt<12>(N1C->getSExtValue()) || !N0.hasOneUse()))
      break;
    // If this can be a ZEXT.w, don't do this if the ZEXT has multiple users or
    // the constant is a simm32.
    bool IsZExtW = C2 == UINT64_C(0xFFFFFFFF) && Subtarget->hasStdExtZba();
    // With XTHeadBb, we can use TH.EXTU.
    IsZExtW |= C2 == UINT64_C(0xFFFFFFFF) && Subtarget->hasVendorXTHeadBb();
    if (IsZExtW && (isInt<32>(N1C->getSExtValue()) || !N0.hasOneUse()))
      break;

    // We need to shift left the AND input and C1 by a total of XLen bits.

    // How far left do we need to shift the AND input?
    unsigned XLen = Subtarget->getXLen();
    unsigned LeadingZeros = XLen - llvm::bit_width(C2);

    // The constant gets shifted by the remaining amount unless that would
    // shift bits out.
    uint64_t C1 = N1C->getZExtValue();
    unsigned ConstantShift = XLen - LeadingZeros;
    if (ConstantShift > (XLen - llvm::bit_width(C1)))
      break;

    uint64_t ShiftedC1 = C1 << ConstantShift;
    // If this RV32, we need to sign extend the constant.
    if (XLen == 32)
      ShiftedC1 = SignExtend64<32>(ShiftedC1);

    // Create (mulhu (slli X, lzcnt(C2)), C1 << (XLen - lzcnt(C2))).
    SDNode *Imm = selectImm(CurDAG, DL, VT, ShiftedC1, *Subtarget).getNode();
    SDNode *SLLI =
        CurDAG->getMachineNode(Q1::SLLI, DL, VT, N0.getOperand(0),
                               CurDAG->getTargetConstant(LeadingZeros, DL, VT));
    SDNode *MULHU = CurDAG->getMachineNode(Q1::MULHU, DL, VT,
                                           SDValue(SLLI, 0), SDValue(Imm, 0));
    ReplaceNode(Node, MULHU);
    return;
  }
  case ISD::LOAD: {
    if (tryIndexedLoad(Node))
      return;

    if (Subtarget->hasVendorXCVmem() && !Subtarget->is64Bit()) {
      // We match post-incrementing load here
      LoadSDNode *Load = cast<LoadSDNode>(Node);
      if (Load->getAddressingMode() != ISD::POST_INC)
        break;

      SDValue Chain = Node->getOperand(0);
      SDValue Base = Node->getOperand(1);
      SDValue Offset = Node->getOperand(2);

      bool Simm12 = false;
      bool SignExtend = Load->getExtensionType() == ISD::SEXTLOAD;

      if (auto ConstantOffset = dyn_cast<ConstantSDNode>(Offset)) {
        int ConstantVal = ConstantOffset->getSExtValue();
        Simm12 = isInt<12>(ConstantVal);
        if (Simm12)
          Offset = CurDAG->getTargetConstant(ConstantVal, SDLoc(Offset),
                                             Offset.getValueType());
      }

      unsigned Opcode = 0;
      switch (Load->getMemoryVT().getSimpleVT().SimpleTy) {
      case MVT::i8:
        if (Simm12 && SignExtend)
          Opcode = Q1::CV_LB_ri_inc;
        else if (Simm12 && !SignExtend)
          Opcode = Q1::CV_LBU_ri_inc;
        else if (!Simm12 && SignExtend)
          Opcode = Q1::CV_LB_rr_inc;
        else
          Opcode = Q1::CV_LBU_rr_inc;
        break;
      case MVT::i16:
        if (Simm12 && SignExtend)
          Opcode = Q1::CV_LH_ri_inc;
        else if (Simm12 && !SignExtend)
          Opcode = Q1::CV_LHU_ri_inc;
        else if (!Simm12 && SignExtend)
          Opcode = Q1::CV_LH_rr_inc;
        else
          Opcode = Q1::CV_LHU_rr_inc;
        break;
      case MVT::i32:
        if (Simm12)
          Opcode = Q1::CV_LW_ri_inc;
        else
          Opcode = Q1::CV_LW_rr_inc;
        break;
      default:
        break;
      }
      if (!Opcode)
        break;

      ReplaceNode(Node, CurDAG->getMachineNode(Opcode, DL, XLenVT, XLenVT,
                                               Chain.getSimpleValueType(), Base,
                                               Offset, Chain));
      return;
    }
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = Node->getConstantOperandVal(0);
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;
    case Intrinsic::Q1_vmsgeu:
    case Intrinsic::Q1_vmsge: {
      SDValue Src1 = Node->getOperand(1);
      SDValue Src2 = Node->getOperand(2);
      bool IsUnsigned = IntNo == Intrinsic::Q1_vmsgeu;
      bool IsCmpUnsignedZero = false;
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        int64_t CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16) {
          if (!IsUnsigned || CVal != 0)
            break;
          IsCmpUnsignedZero = true;
        }
      }
      MVT Src1VT = Src1.getSimpleValueType();
      unsigned VMSLTOpcode, VMNANDOpcode, VMSetOpcode;
      switch (Q1TargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMSLT_VMNAND_VMSET_OPCODES(lmulenum, suffix, suffix_b)            \
  case Q1II::VLMUL::lmulenum:                                               \
    VMSLTOpcode = IsUnsigned ? Q1::PseudoVMSLTU_VX_##suffix                 \
                             : Q1::PseudoVMSLT_VX_##suffix;                 \
    VMNANDOpcode = Q1::PseudoVMNAND_MM_##suffix;                            \
    VMSetOpcode = Q1::PseudoVMSET_M_##suffix_b;                             \
    break;
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_F8, MF8, B1)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_F4, MF4, B2)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_F2, MF2, B4)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_1, M1, B8)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_2, M2, B16)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_4, M4, B32)
        CASE_VMSLT_VMNAND_VMSET_OPCODES(LMUL_8, M8, B64)
#undef CASE_VMSLT_VMNAND_VMSET_OPCODES
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(3), VL);

      // If vmsgeu with 0 immediate, expand it to vmset.
      if (IsCmpUnsignedZero) {
        ReplaceNode(Node, CurDAG->getMachineNode(VMSetOpcode, DL, VT, VL, SEW));
        return;
      }

      // Expand to
      // vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd
      SDValue Cmp = SDValue(
          CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
          0);
      ReplaceNode(Node, CurDAG->getMachineNode(VMNANDOpcode, DL, VT,
                                               {Cmp, Cmp, VL, SEW}));
      return;
    }
    case Intrinsic::Q1_vmsgeu_mask:
    case Intrinsic::Q1_vmsge_mask: {
      SDValue Src1 = Node->getOperand(2);
      SDValue Src2 = Node->getOperand(3);
      bool IsUnsigned = IntNo == Intrinsic::Q1_vmsgeu_mask;
      bool IsCmpUnsignedZero = false;
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        int64_t CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16) {
          if (!IsUnsigned || CVal != 0)
            break;
          IsCmpUnsignedZero = true;
        }
      }
      MVT Src1VT = Src1.getSimpleValueType();
      unsigned VMSLTOpcode, VMSLTMaskOpcode, VMXOROpcode, VMANDNOpcode,
          VMOROpcode;
      switch (Q1TargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMSLT_OPCODES(lmulenum, suffix, suffix_b)                         \
  case Q1II::VLMUL::lmulenum:                                               \
    VMSLTOpcode = IsUnsigned ? Q1::PseudoVMSLTU_VX_##suffix                 \
                             : Q1::PseudoVMSLT_VX_##suffix;                 \
    VMSLTMaskOpcode = IsUnsigned ? Q1::PseudoVMSLTU_VX_##suffix##_MASK      \
                                 : Q1::PseudoVMSLT_VX_##suffix##_MASK;      \
    break;
        CASE_VMSLT_OPCODES(LMUL_F8, MF8, B1)
        CASE_VMSLT_OPCODES(LMUL_F4, MF4, B2)
        CASE_VMSLT_OPCODES(LMUL_F2, MF2, B4)
        CASE_VMSLT_OPCODES(LMUL_1, M1, B8)
        CASE_VMSLT_OPCODES(LMUL_2, M2, B16)
        CASE_VMSLT_OPCODES(LMUL_4, M4, B32)
        CASE_VMSLT_OPCODES(LMUL_8, M8, B64)
#undef CASE_VMSLT_OPCODES
      }
      // Mask operations use the LMUL from the mask type.
      switch (Q1TargetLowering::getLMUL(VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMXOR_VMANDN_VMOR_OPCODES(lmulenum, suffix)                       \
  case Q1II::VLMUL::lmulenum:                                               \
    VMXOROpcode = Q1::PseudoVMXOR_MM_##suffix;                              \
    VMANDNOpcode = Q1::PseudoVMANDN_MM_##suffix;                            \
    VMOROpcode = Q1::PseudoVMOR_MM_##suffix;                                \
    break;
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F8, MF8)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F4, MF4)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F2, MF2)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_1, M1)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_2, M2)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_4, M4)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_8, M8)
#undef CASE_VMXOR_VMANDN_VMOR_OPCODES
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue MaskSEW = CurDAG->getTargetConstant(0, DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(5), VL);
      SDValue MaskedOff = Node->getOperand(1);
      SDValue Mask = Node->getOperand(4);

      // If vmsgeu_mask with 0 immediate, expand it to vmor mask, maskedoff.
      if (IsCmpUnsignedZero) {
        // We don't need vmor if the MaskedOff and the Mask are the same
        // value.
        if (Mask == MaskedOff) {
          ReplaceUses(Node, Mask.getNode());
          return;
        }
        ReplaceNode(Node,
                    CurDAG->getMachineNode(VMOROpcode, DL, VT,
                                           {Mask, MaskedOff, VL, MaskSEW}));
        return;
      }

      // If the MaskedOff value and the Mask are the same value use
      // vmslt{u}.vx vt, va, x;  vmandn.mm vd, vd, vt
      // This avoids needing to copy v0 to vd before starting the next sequence.
      if (Mask == MaskedOff) {
        SDValue Cmp = SDValue(
            CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
            0);
        ReplaceNode(Node, CurDAG->getMachineNode(VMANDNOpcode, DL, VT,
                                                 {Mask, Cmp, VL, MaskSEW}));
        return;
      }

      // Mask needs to be copied to V0.
      SDValue Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), DL,
                                           Q1::V0, Mask, SDValue());
      SDValue Glue = Chain.getValue(1);
      SDValue V0 = CurDAG->getRegister(Q1::V0, VT);

      // Otherwise use
      // vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0
      // The result is mask undisturbed.
      // We use the same instructions to emulate mask agnostic behavior, because
      // the agnostic result can be either undisturbed or all 1.
      SDValue Cmp = SDValue(
          CurDAG->getMachineNode(VMSLTMaskOpcode, DL, VT,
                                 {MaskedOff, Src1, Src2, V0, VL, SEW, Glue}),
          0);
      // vmxor.mm vd, vd, v0 is used to update active value.
      ReplaceNode(Node, CurDAG->getMachineNode(VMXOROpcode, DL, VT,
                                               {Cmp, Mask, VL, MaskSEW}));
      return;
    }
    case Intrinsic::Q1_vsetvli:
    case Intrinsic::Q1_vsetvlimax:
      return selectVSETVLI(Node);
    }
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = Node->getConstantOperandVal(1);
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;
    case Intrinsic::Q1_vlseg2:
    case Intrinsic::Q1_vlseg3:
    case Intrinsic::Q1_vlseg4:
    case Intrinsic::Q1_vlseg5:
    case Intrinsic::Q1_vlseg6:
    case Intrinsic::Q1_vlseg7:
    case Intrinsic::Q1_vlseg8: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::Q1_vlseg2_mask:
    case Intrinsic::Q1_vlseg3_mask:
    case Intrinsic::Q1_vlseg4_mask:
    case Intrinsic::Q1_vlseg5_mask:
    case Intrinsic::Q1_vlseg6_mask:
    case Intrinsic::Q1_vlseg7_mask:
    case Intrinsic::Q1_vlseg8_mask: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::Q1_vlsseg2:
    case Intrinsic::Q1_vlsseg3:
    case Intrinsic::Q1_vlsseg4:
    case Intrinsic::Q1_vlsseg5:
    case Intrinsic::Q1_vlsseg6:
    case Intrinsic::Q1_vlsseg7:
    case Intrinsic::Q1_vlsseg8: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::Q1_vlsseg2_mask:
    case Intrinsic::Q1_vlsseg3_mask:
    case Intrinsic::Q1_vlsseg4_mask:
    case Intrinsic::Q1_vlsseg5_mask:
    case Intrinsic::Q1_vlsseg6_mask:
    case Intrinsic::Q1_vlsseg7_mask:
    case Intrinsic::Q1_vlsseg8_mask: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::Q1_vloxseg2:
    case Intrinsic::Q1_vloxseg3:
    case Intrinsic::Q1_vloxseg4:
    case Intrinsic::Q1_vloxseg5:
    case Intrinsic::Q1_vloxseg6:
    case Intrinsic::Q1_vloxseg7:
    case Intrinsic::Q1_vloxseg8:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::Q1_vluxseg2:
    case Intrinsic::Q1_vluxseg3:
    case Intrinsic::Q1_vluxseg4:
    case Intrinsic::Q1_vluxseg5:
    case Intrinsic::Q1_vluxseg6:
    case Intrinsic::Q1_vluxseg7:
    case Intrinsic::Q1_vluxseg8:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::Q1_vloxseg2_mask:
    case Intrinsic::Q1_vloxseg3_mask:
    case Intrinsic::Q1_vloxseg4_mask:
    case Intrinsic::Q1_vloxseg5_mask:
    case Intrinsic::Q1_vloxseg6_mask:
    case Intrinsic::Q1_vloxseg7_mask:
    case Intrinsic::Q1_vloxseg8_mask:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::Q1_vluxseg2_mask:
    case Intrinsic::Q1_vluxseg3_mask:
    case Intrinsic::Q1_vluxseg4_mask:
    case Intrinsic::Q1_vluxseg5_mask:
    case Intrinsic::Q1_vluxseg6_mask:
    case Intrinsic::Q1_vluxseg7_mask:
    case Intrinsic::Q1_vluxseg8_mask:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::Q1_vlseg8ff:
    case Intrinsic::Q1_vlseg7ff:
    case Intrinsic::Q1_vlseg6ff:
    case Intrinsic::Q1_vlseg5ff:
    case Intrinsic::Q1_vlseg4ff:
    case Intrinsic::Q1_vlseg3ff:
    case Intrinsic::Q1_vlseg2ff: {
      selectVLSEGFF(Node, getSegInstNF(IntNo), /*IsMasked*/ false);
      return;
    }
    case Intrinsic::Q1_vlseg8ff_mask:
    case Intrinsic::Q1_vlseg7ff_mask:
    case Intrinsic::Q1_vlseg6ff_mask:
    case Intrinsic::Q1_vlseg5ff_mask:
    case Intrinsic::Q1_vlseg4ff_mask:
    case Intrinsic::Q1_vlseg3ff_mask:
    case Intrinsic::Q1_vlseg2ff_mask: {
      selectVLSEGFF(Node, getSegInstNF(IntNo), /*IsMasked*/ true);
      return;
    }
    case Intrinsic::Q1_vloxei:
    case Intrinsic::Q1_vloxei_mask:
    case Intrinsic::Q1_vluxei:
    case Intrinsic::Q1_vluxei_mask: {
      bool IsMasked = IntNo == Intrinsic::Q1_vloxei_mask ||
                      IntNo == Intrinsic::Q1_vluxei_mask;
      bool IsOrdered = IntNo == Intrinsic::Q1_vloxei ||
                       IntNo == Intrinsic::Q1_vloxei_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++));

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 /*IsLoad=*/true, &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);
      Q1II::VLMUL IndexLMUL = Q1TargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
        report_fatal_error("The V extension does not support EEW=64 for index "
                           "values when XLEN=32");
      }
      const Q1::VLX_VSXPseudo *P = Q1::getVLXPseudo(
          IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
          static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::Q1_vlm:
    case Intrinsic::Q1_vle:
    case Intrinsic::Q1_vle_mask:
    case Intrinsic::Q1_vlse:
    case Intrinsic::Q1_vlse_mask: {
      bool IsMasked = IntNo == Intrinsic::Q1_vle_mask ||
                      IntNo == Intrinsic::Q1_vlse_mask;
      bool IsStrided =
          IntNo == Intrinsic::Q1_vlse || IntNo == Intrinsic::Q1_vlse_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      // The Q1_vlm intrinsic are always tail agnostic and no passthru
      // operand at the IR level.  In pseudos, they have both policy and
      // passthru operand. The passthru operand is needed to track the
      // "tail undefined" state, and the policy is there just for
      // for consistency - it will always be "don't care" for the
      // unmasked form.
      bool HasPassthruOperand = IntNo != Intrinsic::Q1_vlm;
      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      if (HasPassthruOperand)
        Operands.push_back(Node->getOperand(CurOp++));
      else {
        // We eagerly lower to implicit_def (instead of undef), as we
        // otherwise fail to select nodes such as: nxv1i1 = undef
        SDNode *Passthru =
          CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, VT);
        Operands.push_back(SDValue(Passthru, 0));
      }
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands, /*IsLoad=*/true);

      Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);
      const Q1::VLEPseudo *P =
          Q1::getVLEPseudo(IsMasked, IsStrided, /*FF*/ false, Log2SEW,
                              static_cast<unsigned>(LMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::Q1_vleff:
    case Intrinsic::Q1_vleff_mask: {
      bool IsMasked = IntNo == Intrinsic::Q1_vleff_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 7> Operands;
      Operands.push_back(Node->getOperand(CurOp++));
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ false, Operands,
                                 /*IsLoad=*/true);

      Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);
      const Q1::VLEPseudo *P =
          Q1::getVLEPseudo(IsMasked, /*Strided*/ false, /*FF*/ true,
                              Log2SEW, static_cast<unsigned>(LMUL));
      MachineSDNode *Load = CurDAG->getMachineNode(
          P->Pseudo, DL, Node->getVTList(), Operands);
      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_VOID: {
    unsigned IntNo = Node->getConstantOperandVal(1);
    switch (IntNo) {
    case Intrinsic::Q1_vsseg2:
    case Intrinsic::Q1_vsseg3:
    case Intrinsic::Q1_vsseg4:
    case Intrinsic::Q1_vsseg5:
    case Intrinsic::Q1_vsseg6:
    case Intrinsic::Q1_vsseg7:
    case Intrinsic::Q1_vsseg8: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::Q1_vsseg2_mask:
    case Intrinsic::Q1_vsseg3_mask:
    case Intrinsic::Q1_vsseg4_mask:
    case Intrinsic::Q1_vsseg5_mask:
    case Intrinsic::Q1_vsseg6_mask:
    case Intrinsic::Q1_vsseg7_mask:
    case Intrinsic::Q1_vsseg8_mask: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::Q1_vssseg2:
    case Intrinsic::Q1_vssseg3:
    case Intrinsic::Q1_vssseg4:
    case Intrinsic::Q1_vssseg5:
    case Intrinsic::Q1_vssseg6:
    case Intrinsic::Q1_vssseg7:
    case Intrinsic::Q1_vssseg8: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::Q1_vssseg2_mask:
    case Intrinsic::Q1_vssseg3_mask:
    case Intrinsic::Q1_vssseg4_mask:
    case Intrinsic::Q1_vssseg5_mask:
    case Intrinsic::Q1_vssseg6_mask:
    case Intrinsic::Q1_vssseg7_mask:
    case Intrinsic::Q1_vssseg8_mask: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::Q1_vsoxseg2:
    case Intrinsic::Q1_vsoxseg3:
    case Intrinsic::Q1_vsoxseg4:
    case Intrinsic::Q1_vsoxseg5:
    case Intrinsic::Q1_vsoxseg6:
    case Intrinsic::Q1_vsoxseg7:
    case Intrinsic::Q1_vsoxseg8:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::Q1_vsuxseg2:
    case Intrinsic::Q1_vsuxseg3:
    case Intrinsic::Q1_vsuxseg4:
    case Intrinsic::Q1_vsuxseg5:
    case Intrinsic::Q1_vsuxseg6:
    case Intrinsic::Q1_vsuxseg7:
    case Intrinsic::Q1_vsuxseg8:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::Q1_vsoxseg2_mask:
    case Intrinsic::Q1_vsoxseg3_mask:
    case Intrinsic::Q1_vsoxseg4_mask:
    case Intrinsic::Q1_vsoxseg5_mask:
    case Intrinsic::Q1_vsoxseg6_mask:
    case Intrinsic::Q1_vsoxseg7_mask:
    case Intrinsic::Q1_vsoxseg8_mask:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::Q1_vsuxseg2_mask:
    case Intrinsic::Q1_vsuxseg3_mask:
    case Intrinsic::Q1_vsuxseg4_mask:
    case Intrinsic::Q1_vsuxseg5_mask:
    case Intrinsic::Q1_vsuxseg6_mask:
    case Intrinsic::Q1_vsuxseg7_mask:
    case Intrinsic::Q1_vsuxseg8_mask:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::Q1_vsoxei:
    case Intrinsic::Q1_vsoxei_mask:
    case Intrinsic::Q1_vsuxei:
    case Intrinsic::Q1_vsuxei_mask: {
      bool IsMasked = IntNo == Intrinsic::Q1_vsoxei_mask ||
                      IntNo == Intrinsic::Q1_vsuxei_mask;
      bool IsOrdered = IntNo == Intrinsic::Q1_vsoxei ||
                       IntNo == Intrinsic::Q1_vsoxei_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 /*IsLoad=*/false, &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);
      Q1II::VLMUL IndexLMUL = Q1TargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
        report_fatal_error("The V extension does not support EEW=64 for index "
                           "values when XLEN=32");
      }
      const Q1::VLX_VSXPseudo *P = Q1::getVSXPseudo(
          IsMasked, IsOrdered, IndexLog2EEW,
          static_cast<unsigned>(LMUL), static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    case Intrinsic::Q1_vsm:
    case Intrinsic::Q1_vse:
    case Intrinsic::Q1_vse_mask:
    case Intrinsic::Q1_vsse:
    case Intrinsic::Q1_vsse_mask: {
      bool IsMasked = IntNo == Intrinsic::Q1_vse_mask ||
                      IntNo == Intrinsic::Q1_vsse_mask;
      bool IsStrided =
          IntNo == Intrinsic::Q1_vsse || IntNo == Intrinsic::Q1_vsse_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands);

      Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);
      const Q1::VSEPseudo *P = Q1::getVSEPseudo(
          IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);
      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Store, {MemOp->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    case Intrinsic::Q1_sf_vc_x_se:
    case Intrinsic::Q1_sf_vc_i_se:
      selectSF_VC_X_SE(Node);
      return;
    }
    break;
  }
  case ISD::BITCAST: {
    MVT SrcVT = Node->getOperand(0).getSimpleValueType();
    // Just drop bitcasts between vectors if both are fixed or both are
    // scalable.
    if ((VT.isScalableVector() && SrcVT.isScalableVector()) ||
        (VT.isFixedLengthVector() && SrcVT.isFixedLengthVector())) {
      ReplaceUses(SDValue(Node, 0), Node->getOperand(0));
      CurDAG->RemoveDeadNode(Node);
      return;
    }
    break;
  }
  case ISD::INSERT_SUBVECTOR:
  case Q1ISD::TUPLE_INSERT: {
    SDValue V = Node->getOperand(0);
    SDValue SubV = Node->getOperand(1);
    SDLoc DL(SubV);
    auto Idx = Node->getConstantOperandVal(2);
    MVT SubVecVT = SubV.getSimpleValueType();

    const Q1TargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = SubVecVT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (SubVecVT.isFixedLengthVector()) {
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(SubVecVT);
      TypeSize VecRegSize = TypeSize::getScalable(Q1::RVVBitsPerBlock);
      [[maybe_unused]] bool ExactlyVecRegSized =
          Subtarget->expandVScale(SubVecVT.getSizeInBits())
              .isKnownMultipleOf(Subtarget->expandVScale(VecRegSize));
      assert(isPowerOf2_64(Subtarget->expandVScale(SubVecVT.getSizeInBits())
                               .getKnownMinValue()));
      assert(Idx == 0 && (ExactlyVecRegSized || V.isUndef()));
    }
    MVT ContainerVT = VT;
    if (VT.isFixedLengthVector())
      ContainerVT = TLI.getContainerForFixedLengthVector(VT);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        Q1TargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            ContainerVT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // insert which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    Q1II::VLMUL SubVecLMUL = Q1TargetLowering::getLMUL(SubVecContainerVT);
    [[maybe_unused]] bool IsSubVecPartReg =
        SubVecLMUL == Q1II::VLMUL::LMUL_F2 ||
        SubVecLMUL == Q1II::VLMUL::LMUL_F4 ||
        SubVecLMUL == Q1II::VLMUL::LMUL_F8;
    assert((V.getValueType().isQ1VectorTuple() || !IsSubVecPartReg ||
            V.isUndef()) &&
           "Expecting lowering to have created legal INSERT_SUBVECTORs when "
           "the subvector is smaller than a full-sized register");

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL groups (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == Q1::NoSubRegister) {
      unsigned InRegClassID =
          Q1TargetLowering::getRegClassIDForVecVT(ContainerVT);
      assert(Q1TargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                               DL, VT, SubV, RC);
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Insert = CurDAG->getTargetInsertSubreg(SubRegIdx, DL, VT, V, SubV);
    ReplaceNode(Node, Insert.getNode());
    return;
  }
  case ISD::EXTRACT_SUBVECTOR:
  case Q1ISD::TUPLE_EXTRACT: {
    SDValue V = Node->getOperand(0);
    auto Idx = Node->getConstantOperandVal(1);
    MVT InVT = V.getSimpleValueType();
    SDLoc DL(V);

    const Q1TargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = VT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (VT.isFixedLengthVector()) {
      assert(Idx == 0);
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(VT);
    }
    if (InVT.isFixedLengthVector())
      InVT = TLI.getContainerForFixedLengthVector(InVT);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        Q1TargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            InVT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // extract which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL types (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == Q1::NoSubRegister) {
      unsigned InRegClassID = Q1TargetLowering::getRegClassIDForVecVT(InVT);
      assert(Q1TargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode =
          CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, DL, VT, V, RC);
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Extract = CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, V);
    ReplaceNode(Node, Extract.getNode());
    return;
  }
  case Q1ISD::VMV_S_X_VL:
  case Q1ISD::VFMV_S_F_VL:
  case Q1ISD::VMV_V_X_VL:
  case Q1ISD::VFMV_V_F_VL: {
    // Try to match splat of a scalar load to a strided load with stride of x0.
    bool IsScalarMove = Node->getOpcode() == Q1ISD::VMV_S_X_VL ||
                        Node->getOpcode() == Q1ISD::VFMV_S_F_VL;
    if (!Node->getOperand(0).isUndef())
      break;
    SDValue Src = Node->getOperand(1);
    auto *Ld = dyn_cast<LoadSDNode>(Src);
    // Can't fold load update node because the second
    // output is used so that load update node can't be removed.
    if (!Ld || Ld->isIndexed())
      break;
    EVT MemVT = Ld->getMemoryVT();
    // The memory VT should be the same size as the element type.
    if (MemVT.getStoreSize() != VT.getVectorElementType().getStoreSize())
      break;
    if (!IsProfitableToFold(Src, Node, Node) ||
        !IsLegalToFold(Src, Node, Node, TM.getOptLevel()))
      break;

    SDValue VL;
    if (IsScalarMove) {
      // We could deal with more VL if we update the VSETVLI insert pass to
      // avoid introducing more VSETVLI.
      if (!isOneConstant(Node->getOperand(2)))
        break;
      selectVLOp(Node->getOperand(2), VL);
    } else
      selectVLOp(Node->getOperand(2), VL);

    unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
    SDValue SEW = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);

    // If VL=1, then we don't need to do a strided load and can just do a
    // regular load.
    bool IsStrided = !isOneConstant(VL);

    // Only do a strided load if we have optimized zero-stride vector load.
    if (IsStrided && !Subtarget->hasOptimizedZeroStrideLoad())
      break;

    SmallVector<SDValue> Operands = {
        SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, VT), 0),
        Ld->getBasePtr()};
    if (IsStrided)
      Operands.push_back(CurDAG->getRegister(Q1::X0, XLenVT));
    uint64_t Policy = Q1II::MASK_AGNOSTIC | Q1II::TAIL_AGNOSTIC;
    SDValue PolicyOp = CurDAG->getTargetConstant(Policy, DL, XLenVT);
    Operands.append({VL, SEW, PolicyOp, Ld->getChain()});

    Q1II::VLMUL LMUL = Q1TargetLowering::getLMUL(VT);
    const Q1::VLEPseudo *P = Q1::getVLEPseudo(
        /*IsMasked*/ false, IsStrided, /*FF*/ false,
        Log2SEW, static_cast<unsigned>(LMUL));
    MachineSDNode *Load =
        CurDAG->getMachineNode(P->Pseudo, DL, {VT, MVT::Other}, Operands);
    // Update the chain.
    ReplaceUses(Src.getValue(1), SDValue(Load, 1));
    // Record the mem-refs
    CurDAG->setNodeMemRefs(Load, {Ld->getMemOperand()});
    // Replace the splat with the vlse.
    ReplaceNode(Node, Load);
    return;
  }
  case ISD::PREFETCH:
    unsigned Locality = Node->getConstantOperandVal(3);
    if (Locality > 2)
      break;

    if (auto *LoadStoreMem = dyn_cast<MemSDNode>(Node)) {
      MachineMemOperand *MMO = LoadStoreMem->getMemOperand();
      MMO->setFlags(MachineMemOperand::MONonTemporal);

      int NontemporalLevel = 0;
      switch (Locality) {
      case 0:
        NontemporalLevel = 3; // NTL.ALL
        break;
      case 1:
        NontemporalLevel = 1; // NTL.PALL
        break;
      case 2:
        NontemporalLevel = 0; // NTL.P1
        break;
      default:
        llvm_unreachable("unexpected locality value.");
      }

      if (NontemporalLevel & 0b1)
        MMO->setFlags(MONontemporalBit0);
      if (NontemporalLevel & 0b10)
        MMO->setFlags(MONontemporalBit1);
    }
    break;
  }

  // Select the default instruction.
  SelectCode(Node);
}

bool Q1DAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
    std::vector<SDValue> &OutOps) {
  // Always produce a register and immediate operand, as expected by
  // Q1AsmPrinter::PrintAsmMemoryOperand.
  switch (ConstraintID) {
  case InlineAsm::ConstraintCode::o:
  case InlineAsm::ConstraintCode::m: {
    SDValue Op0, Op1;
    [[maybe_unused]] bool Found = SelectAddrRegImm(Op, Op0, Op1);
    assert(Found && "SelectAddrRegImm should always succeed");
    OutOps.push_back(Op0);
    OutOps.push_back(Op1);
    return false;
  }
  case InlineAsm::ConstraintCode::A:
    OutOps.push_back(Op);
    OutOps.push_back(
        CurDAG->getTargetConstant(0, SDLoc(Op), Subtarget->getXLenVT()));
    return false;
  default:
    report_fatal_error("Unexpected asm memory constraint " +
                       InlineAsm::getMemConstraintName(ConstraintID));
  }

  return true;
}

bool Q1DAGToDAGISel::SelectAddrFrameIndex(SDValue Addr, SDValue &Base,
                                             SDValue &Offset) {
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), Subtarget->getXLenVT());
    return true;
  }

  return false;
}

// Select a frame index and an optional immediate offset from an ADD or OR.
bool Q1DAGToDAGISel::SelectFrameAddrRegImm(SDValue Addr, SDValue &Base,
                                              SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (!CurDAG->isBaseWithConstantOffset(Addr))
    return false;

  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr.getOperand(0))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal)) {
      Base = CurDAG->getTargetFrameIndex(FIN->getIndex(),
                                         Subtarget->getXLenVT());
      Offset = CurDAG->getSignedConstant(
          CVal, SDLoc(Addr), Subtarget->getXLenVT(), /*isTarget=*/true);
      return true;
    }
  }

  return false;
}

// Fold constant addresses.
static bool selectConstantAddr(SelectionDAG *CurDAG, const SDLoc &DL,
                               const MVT VT, const Q1Subtarget *Subtarget,
                               SDValue Addr, SDValue &Base, SDValue &Offset,
                               bool IsPrefetch = false,
                               bool IsRV32Zdinx = false) {
  if (!isa<ConstantSDNode>(Addr))
    return false;

  int64_t CVal = cast<ConstantSDNode>(Addr)->getSExtValue();

  // If the constant is a simm12, we can fold the whole constant and use X0 as
  // the base. If the constant can be materialized with LUI+simm12, use LUI as
  // the base. We can't use generateInstSeq because it favors LUI+ADDIW.
  int64_t Lo12 = SignExtend64<12>(CVal);
  int64_t Hi = (uint64_t)CVal - (uint64_t)Lo12;
  if (!Subtarget->is64Bit() || isInt<32>(Hi)) {
    if (IsPrefetch && (Lo12 & 0b11111) != 0)
      return false;
    if (IsRV32Zdinx && !isInt<12>(Lo12 + 4))
      return false;

    if (Hi) {
      int64_t Hi20 = (Hi >> 12) & 0xfffff;
      Base = SDValue(
          CurDAG->getMachineNode(Q1::LUI, DL, VT,
                                 CurDAG->getTargetConstant(Hi20, DL, VT)),
          0);
    } else {
      Base = CurDAG->getRegister(Q1::X0, VT);
    }
    Offset = CurDAG->getSignedConstant(Lo12, DL, VT, /*isTarget=*/true);
    return true;
  }

  // Ask how constant materialization would handle this constant.
  Q1MatInt::InstSeq Seq = Q1MatInt::generateInstSeq(CVal, *Subtarget);

  // If the last instruction would be an ADDI, we can fold its immediate and
  // emit the rest of the sequence as the base.
  if (Seq.back().getOpcode() != Q1::ADDI)
    return false;
  Lo12 = Seq.back().getImm();
  if (IsPrefetch && (Lo12 & 0b11111) != 0)
    return false;
  if (IsRV32Zdinx && !isInt<12>(Lo12 + 4))
    return false;

  // Drop the last instruction.
  Seq.pop_back();
  assert(!Seq.empty() && "Expected more instructions in sequence");

  Base = selectImmSeq(CurDAG, DL, VT, Seq);
  Offset = CurDAG->getSignedConstant(Lo12, DL, VT, /*isTarget=*/true);
  return true;
}

// Is this ADD instruction only used as the base pointer of scalar loads and
// stores?
static bool isWorthFoldingAdd(SDValue Add) {
  for (auto *Use : Add->uses()) {
    if (Use->getOpcode() != ISD::LOAD && Use->getOpcode() != ISD::STORE &&
        Use->getOpcode() != ISD::ATOMIC_LOAD &&
        Use->getOpcode() != ISD::ATOMIC_STORE)
      return false;
    EVT VT = cast<MemSDNode>(Use)->getMemoryVT();
    if (!VT.isScalarInteger() && VT != MVT::f16 && VT != MVT::f32 &&
        VT != MVT::f64)
      return false;
    // Don't allow stores of the value. It must be used as the address.
    if (Use->getOpcode() == ISD::STORE &&
        cast<StoreSDNode>(Use)->getValue() == Add)
      return false;
    if (Use->getOpcode() == ISD::ATOMIC_STORE &&
        cast<AtomicSDNode>(Use)->getVal() == Add)
      return false;
  }

  return true;
}

bool Q1DAGToDAGISel::SelectAddrRegRegScale(SDValue Addr,
                                              unsigned MaxShiftAmount,
                                              SDValue &Base, SDValue &Index,
                                              SDValue &Scale) {
  EVT VT = Addr.getSimpleValueType();
  auto UnwrapShl = [this, VT, MaxShiftAmount](SDValue N, SDValue &Index,
                                              SDValue &Shift) {
    uint64_t ShiftAmt = 0;
    Index = N;

    if (N.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N.getOperand(1))) {
      // Only match shifts by a value in range [0, MaxShiftAmount].
      if (N.getConstantOperandVal(1) <= MaxShiftAmount) {
        Index = N.getOperand(0);
        ShiftAmt = N.getConstantOperandVal(1);
      }
    }

    Shift = CurDAG->getTargetConstant(ShiftAmt, SDLoc(N), VT);
    return ShiftAmt != 0;
  };

  if (Addr.getOpcode() == ISD::ADD) {
    if (auto *C1 = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      SDValue AddrB = Addr.getOperand(0);
      if (AddrB.getOpcode() == ISD::ADD &&
          UnwrapShl(AddrB.getOperand(0), Index, Scale) &&
          !isa<ConstantSDNode>(AddrB.getOperand(1)) &&
          isInt<12>(C1->getSExtValue())) {
        // (add (add (shl A C2) B) C1) -> (add (add B C1) (shl A C2))
        SDValue C1Val =
            CurDAG->getTargetConstant(C1->getZExtValue(), SDLoc(Addr), VT);
        Base = SDValue(CurDAG->getMachineNode(Q1::ADDI, SDLoc(Addr), VT,
                                              AddrB.getOperand(1), C1Val),
                       0);
        return true;
      }
    } else if (UnwrapShl(Addr.getOperand(0), Index, Scale)) {
      Base = Addr.getOperand(1);
      return true;
    } else {
      UnwrapShl(Addr.getOperand(1), Index, Scale);
      Base = Addr.getOperand(0);
      return true;
    }
  } else if (UnwrapShl(Addr, Index, Scale)) {
    EVT VT = Addr.getValueType();
    Base = CurDAG->getRegister(Q1::X0, VT);
    return true;
  }

  return false;
}

bool Q1DAGToDAGISel::SelectAddrRegImm(SDValue Addr, SDValue &Base,
                                         SDValue &Offset, bool IsRV32Zdinx) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (Addr.getOpcode() == Q1ISD::ADD_LO) {
    // If this is non RV32Zdinx we can always fold.
    if (!IsRV32Zdinx) {
      Base = Addr.getOperand(0);
      Offset = Addr.getOperand(1);
      return true;
    }

    // For RV32Zdinx we need to have more than 4 byte alignment so we can add 4
    // to the offset when we expand in Q1ExpandPseudoInsts.
    if (auto *GA = dyn_cast<GlobalAddressSDNode>(Addr.getOperand(1))) {
      const DataLayout &DL = CurDAG->getDataLayout();
      Align Alignment = commonAlignment(
          GA->getGlobal()->getPointerAlignment(DL), GA->getOffset());
      if (Alignment > 4) {
        Base = Addr.getOperand(0);
        Offset = Addr.getOperand(1);
        return true;
      }
    }
    if (auto *CP = dyn_cast<ConstantPoolSDNode>(Addr.getOperand(1))) {
      Align Alignment = commonAlignment(CP->getAlign(), CP->getOffset());
      if (Alignment > 4) {
        Base = Addr.getOperand(0);
        Offset = Addr.getOperand(1);
        return true;
      }
    }
  }

  int64_t RV32ZdinxRange = IsRV32Zdinx ? 4 : 0;
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal) && isInt<12>(CVal + RV32ZdinxRange)) {
      Base = Addr.getOperand(0);
      if (Base.getOpcode() == Q1ISD::ADD_LO) {
        SDValue LoOperand = Base.getOperand(1);
        if (auto *GA = dyn_cast<GlobalAddressSDNode>(LoOperand)) {
          // If the Lo in (ADD_LO hi, lo) is a global variable's address
          // (its low part, really), then we can rely on the alignment of that
          // variable to provide a margin of safety before low part can overflow
          // the 12 bits of the load/store offset. Check if CVal falls within
          // that margin; if so (low part + CVal) can't overflow.
          const DataLayout &DL = CurDAG->getDataLayout();
          Align Alignment = commonAlignment(
              GA->getGlobal()->getPointerAlignment(DL), GA->getOffset());
          if ((CVal == 0 || Alignment > CVal) &&
              (!IsRV32Zdinx || commonAlignment(Alignment, CVal) > 4)) {
            int64_t CombinedOffset = CVal + GA->getOffset();
            Base = Base.getOperand(0);
            Offset = CurDAG->getTargetGlobalAddress(
                GA->getGlobal(), SDLoc(LoOperand), LoOperand.getValueType(),
                CombinedOffset, GA->getTargetFlags());
            return true;
          }
        }
      }

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getSignedConstant(CVal, DL, VT, /*isTarget=*/true);
      return true;
    }
  }

  // Handle ADD with large immediates.
  if (Addr.getOpcode() == ISD::ADD && isa<ConstantSDNode>(Addr.getOperand(1))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    assert(!(isInt<12>(CVal) && isInt<12>(CVal + RV32ZdinxRange)) &&
           "simm12 not already handled?");

    // Handle immediates in the range [-4096,-2049] or [2048, 4094]. We can use
    // an ADDI for part of the offset and fold the rest into the load/store.
    // This mirrors the AddiPair PatFrag in Q1InstrInfo.td.
    if (CVal >= -4096 && CVal <= (4094 - RV32ZdinxRange)) {
      int64_t Adj = CVal < 0 ? -2048 : 2047;
      Base = SDValue(
          CurDAG->getMachineNode(
              Q1::ADDI, DL, VT, Addr.getOperand(0),
              CurDAG->getSignedConstant(Adj, DL, VT, /*isTarget=*/true)),
          0);
      Offset = CurDAG->getSignedConstant(CVal - Adj, DL, VT, /*isTarget=*/true);
      return true;
    }

    // For larger immediates, we might be able to save one instruction from
    // constant materialization by folding the Lo12 bits of the immediate into
    // the address. We should only do this if the ADD is only used by loads and
    // stores that can fold the lo12 bits. Otherwise, the ADD will get iseled
    // separately with the full materialized immediate creating extra
    // instructions.
    if (isWorthFoldingAdd(Addr) &&
        selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr.getOperand(1), Base,
                           Offset, /*IsPrefetch=*/false, RV32ZdinxRange)) {
      // Insert an ADD instruction with the materialized Hi52 bits.
      Base = SDValue(
          CurDAG->getMachineNode(Q1::ADD, DL, VT, Addr.getOperand(0), Base),
          0);
      return true;
    }
  }

  if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr, Base, Offset,
                         /*IsPrefetch=*/false, RV32ZdinxRange))
    return true;

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}

/// Similar to SelectAddrRegImm, except that the least significant 5 bits of
/// Offset should be all zeros.
bool Q1DAGToDAGISel::SelectAddrRegImmLsb00000(SDValue Addr, SDValue &Base,
                                                 SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal)) {
      Base = Addr.getOperand(0);

      // Early-out if not a valid offset.
      if ((CVal & 0b11111) != 0) {
        Base = Addr;
        Offset = CurDAG->getTargetConstant(0, DL, VT);
        return true;
      }

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getSignedConstant(CVal, DL, VT, /*isTarget=*/true);
      return true;
    }
  }

  // Handle ADD with large immediates.
  if (Addr.getOpcode() == ISD::ADD && isa<ConstantSDNode>(Addr.getOperand(1))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    assert(!(isInt<12>(CVal) && isInt<12>(CVal)) &&
           "simm12 not already handled?");

    // Handle immediates in the range [-4096,-2049] or [2017, 4065]. We can save
    // one instruction by folding adjustment (-2048 or 2016) into the address.
    if ((-2049 >= CVal && CVal >= -4096) || (4065 >= CVal && CVal >= 2017)) {
      int64_t Adj = CVal < 0 ? -2048 : 2016;
      int64_t AdjustedOffset = CVal - Adj;
      Base = SDValue(CurDAG->getMachineNode(
                         Q1::ADDI, DL, VT, Addr.getOperand(0),
                         CurDAG->getSignedConstant(AdjustedOffset, DL, VT,
                                                   /*isTarget=*/true)),
                     0);
      Offset = CurDAG->getSignedConstant(Adj, DL, VT, /*isTarget=*/true);
      return true;
    }

    if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr.getOperand(1), Base,
                           Offset, /*IsPrefetch=*/true)) {
      // Insert an ADD instruction with the materialized Hi52 bits.
      Base = SDValue(
          CurDAG->getMachineNode(Q1::ADD, DL, VT, Addr.getOperand(0), Base),
          0);
      return true;
    }
  }

  if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr, Base, Offset,
                         /*IsPrefetch=*/true))
    return true;

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}

bool Q1DAGToDAGISel::SelectAddrRegReg(SDValue Addr, SDValue &Base,
                                         SDValue &Offset) {
  if (Addr.getOpcode() != ISD::ADD)
    return false;

  if (isa<ConstantSDNode>(Addr.getOperand(1)))
    return false;

  Base = Addr.getOperand(1);
  Offset = Addr.getOperand(0);
  return true;
}

bool Q1DAGToDAGISel::selectShiftMask(SDValue N, unsigned ShiftWidth,
                                        SDValue &ShAmt) {
  ShAmt = N;

  // Peek through zext.
  if (ShAmt->getOpcode() == ISD::ZERO_EXTEND)
    ShAmt = ShAmt.getOperand(0);

  // Shift instructions on RISC-V only read the lower 5 or 6 bits of the shift
  // amount. If there is an AND on the shift amount, we can bypass it if it
  // doesn't affect any of those bits.
  if (ShAmt.getOpcode() == ISD::AND &&
      isa<ConstantSDNode>(ShAmt.getOperand(1))) {
    const APInt &AndMask = ShAmt.getConstantOperandAPInt(1);

    // Since the max shift amount is a power of 2 we can subtract 1 to make a
    // mask that covers the bits needed to represent all shift amounts.
    assert(isPowerOf2_32(ShiftWidth) && "Unexpected max shift amount!");
    APInt ShMask(AndMask.getBitWidth(), ShiftWidth - 1);

    if (ShMask.isSubsetOf(AndMask)) {
      ShAmt = ShAmt.getOperand(0);
    } else {
      // SimplifyDemandedBits may have optimized the mask so try restoring any
      // bits that are known zero.
      KnownBits Known = CurDAG->computeKnownBits(ShAmt.getOperand(0));
      if (!ShMask.isSubsetOf(AndMask | Known.Zero))
        return true;
      ShAmt = ShAmt.getOperand(0);
    }
  }

  if (ShAmt.getOpcode() == ISD::ADD &&
      isa<ConstantSDNode>(ShAmt.getOperand(1))) {
    uint64_t Imm = ShAmt.getConstantOperandVal(1);
    // If we are shifting by X+N where N == 0 mod Size, then just shift by X
    // to avoid the ADD.
    if (Imm != 0 && Imm % ShiftWidth == 0) {
      ShAmt = ShAmt.getOperand(0);
      return true;
    }
  } else if (ShAmt.getOpcode() == ISD::SUB &&
             isa<ConstantSDNode>(ShAmt.getOperand(0))) {
    uint64_t Imm = ShAmt.getConstantOperandVal(0);
    // If we are shifting by N-X where N == 0 mod Size, then just shift by -X to
    // generate a NEG instead of a SUB of a constant.
    if (Imm != 0 && Imm % ShiftWidth == 0) {
      SDLoc DL(ShAmt);
      EVT VT = ShAmt.getValueType();
      SDValue Zero = CurDAG->getRegister(Q1::X0, VT);
      unsigned NegOpc = VT == MVT::i64 ? Q1::SUBW : Q1::SUB;
      MachineSDNode *Neg = CurDAG->getMachineNode(NegOpc, DL, VT, Zero,
                                                  ShAmt.getOperand(1));
      ShAmt = SDValue(Neg, 0);
      return true;
    }
    // If we are shifting by N-X where N == -1 mod Size, then just shift by ~X
    // to generate a NOT instead of a SUB of a constant.
    if (Imm % ShiftWidth == ShiftWidth - 1) {
      SDLoc DL(ShAmt);
      EVT VT = ShAmt.getValueType();
      MachineSDNode *Not = CurDAG->getMachineNode(
          Q1::XORI, DL, VT, ShAmt.getOperand(1),
          CurDAG->getAllOnesConstant(DL, VT, /*isTarget=*/true));
      ShAmt = SDValue(Not, 0);
      return true;
    }
  }

  return true;
}

/// RISC-V doesn't have general instructions for integer setne/seteq, but we can
/// check for equality with 0. This function emits instructions that convert the
/// seteq/setne into something that can be compared with 0.
/// \p ExpectedCCVal indicates the condition code to attempt to match (e.g.
/// ISD::SETNE).
bool Q1DAGToDAGISel::selectSETCC(SDValue N, ISD::CondCode ExpectedCCVal,
                                    SDValue &Val) {
  assert(ISD::isIntEqualitySetCC(ExpectedCCVal) &&
         "Unexpected condition code!");

  // We're looking for a setcc.
  if (N->getOpcode() != ISD::SETCC)
    return false;

  // Must be an equality comparison.
  ISD::CondCode CCVal = cast<CondCodeSDNode>(N->getOperand(2))->get();
  if (CCVal != ExpectedCCVal)
    return false;

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if (!LHS.getValueType().isScalarInteger())
    return false;

  // If the RHS side is 0, we don't need any extra instructions, return the LHS.
  if (isNullConstant(RHS)) {
    Val = LHS;
    return true;
  }

  SDLoc DL(N);

  if (auto *C = dyn_cast<ConstantSDNode>(RHS)) {
    int64_t CVal = C->getSExtValue();
    // If the RHS is -2048, we can use xori to produce 0 if the LHS is -2048 and
    // non-zero otherwise.
    if (CVal == -2048) {
      Val = SDValue(CurDAG->getMachineNode(
                        Q1::XORI, DL, N->getValueType(0), LHS,
                        CurDAG->getSignedConstant(CVal, DL, N->getValueType(0),
                                                  /*isTarget=*/true)),
                    0);
      return true;
    }
    // If the RHS is [-2047,2048], we can use addi with -RHS to produce 0 if the
    // LHS is equal to the RHS and non-zero otherwise.
    if (isInt<12>(CVal) || CVal == 2048) {
      Val = SDValue(CurDAG->getMachineNode(
                        Q1::ADDI, DL, N->getValueType(0), LHS,
                        CurDAG->getSignedConstant(-CVal, DL, N->getValueType(0),
                                                  /*isTarget=*/true)),
                    0);
      return true;
    }
    if (isPowerOf2_64(CVal) && Subtarget->hasStdExtZbs()) {
      Val = SDValue(
          CurDAG->getMachineNode(
              Q1::BINVI, DL, N->getValueType(0), LHS,
              CurDAG->getTargetConstant(Log2_64(CVal), DL, N->getValueType(0))),
          0);
      return true;
    }
  }

  // If nothing else we can XOR the LHS and RHS to produce zero if they are
  // equal and a non-zero value if they aren't.
  Val = SDValue(
      CurDAG->getMachineNode(Q1::XOR, DL, N->getValueType(0), LHS, RHS), 0);
  return true;
}

bool Q1DAGToDAGISel::selectSExtBits(SDValue N, unsigned Bits, SDValue &Val) {
  if (N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      cast<VTSDNode>(N.getOperand(1))->getVT().getSizeInBits() == Bits) {
    Val = N.getOperand(0);
    return true;
  }

  auto UnwrapShlSra = [](SDValue N, unsigned ShiftAmt) {
    if (N.getOpcode() != ISD::SRA || !isa<ConstantSDNode>(N.getOperand(1)))
      return N;

    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N.getConstantOperandVal(1) == ShiftAmt &&
        N0.getConstantOperandVal(1) == ShiftAmt)
      return N0.getOperand(0);

    return N;
  };

  MVT VT = N.getSimpleValueType();
  if (CurDAG->ComputeNumSignBits(N) > (VT.getSizeInBits() - Bits)) {
    Val = UnwrapShlSra(N, VT.getSizeInBits() - Bits);
    return true;
  }

  return false;
}

bool Q1DAGToDAGISel::selectZExtBits(SDValue N, unsigned Bits, SDValue &Val) {
  if (N.getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (C && C->getZExtValue() == maskTrailingOnes<uint64_t>(Bits)) {
      Val = N.getOperand(0);
      return true;
    }
  }
  MVT VT = N.getSimpleValueType();
  APInt Mask = APInt::getBitsSetFrom(VT.getSizeInBits(), Bits);
  if (CurDAG->MaskedValueIsZero(N, Mask)) {
    Val = N;
    return true;
  }

  return false;
}

/// Look for various patterns that can be done with a SHL that can be folded
/// into a SHXADD. \p ShAmt contains 1, 2, or 3 and is set based on which
/// SHXADD we are trying to match.
bool Q1DAGToDAGISel::selectSHXADDOp(SDValue N, unsigned ShAmt,
                                       SDValue &Val) {
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1))) {
    SDValue N0 = N.getOperand(0);

    if (bool LeftShift = N0.getOpcode() == ISD::SHL;
        (LeftShift || N0.getOpcode() == ISD::SRL) &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      unsigned XLen = Subtarget->getXLen();
      if (LeftShift)
        Mask &= maskTrailingZeros<uint64_t>(C2);
      else
        Mask &= maskTrailingOnes<uint64_t>(XLen - C2);

      // Look for (and (shl y, c2), c1) where c1 is a shifted mask with no
      // leading zeros and c3 trailing zeros. We can use an SRLI by c2+c3
      // followed by a SHXADD with c3 for the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        if (LeftShift && Leading == 0 && C2 < Trailing && Trailing == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Q1::SRLI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing - C2, DL, VT)),
                        0);
          return true;
        }
        // Look for (and (shr y, c2), c1) where c1 is a shifted mask with c2
        // leading zeros and c3 trailing zeros. We can use an SRLI by C3
        // followed by a SHXADD using c3 for the X amount.
        if (!LeftShift && Leading == C2 && Trailing == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(
              CurDAG->getMachineNode(
                  Q1::SRLI, DL, VT, N0.getOperand(0),
                  CurDAG->getTargetConstant(Leading + Trailing, DL, VT)),
              0);
          return true;
        }
      }
    } else if (N0.getOpcode() == ISD::SRA && N0.hasOneUse() &&
               isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      // Look for (and (sra y, c2), c1) where c1 is a shifted mask with c3
      // leading zeros and c4 trailing zeros. If c2 is greater than c3, we can
      // use (srli (srai y, c2 - c3), c3 + c4) followed by a SHXADD with c4 as
      // the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned XLen = Subtarget->getXLen();
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        if (C2 > Leading && Leading > 0 && Trailing == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Q1::SRAI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(C2 - Leading, DL, VT)),
                        0);
          Val = SDValue(CurDAG->getMachineNode(
                            Q1::SRLI, DL, VT, Val,
                            CurDAG->getTargetConstant(Leading + ShAmt, DL, VT)),
                        0);
          return true;
        }
      }
    }
  } else if (bool LeftShift = N.getOpcode() == ISD::SHL;
             (LeftShift || N.getOpcode() == ISD::SRL) &&
             isa<ConstantSDNode>(N.getOperand(1))) {
    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::AND && N0.hasOneUse() &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N0.getConstantOperandVal(1);
      if (isShiftedMask_64(Mask)) {
        unsigned C1 = N.getConstantOperandVal(1);
        unsigned XLen = Subtarget->getXLen();
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        // Look for (shl (and X, Mask), C1) where Mask has 32 leading zeros and
        // C3 trailing zeros. If C1+C3==ShAmt we can use SRLIW+SHXADD.
        if (LeftShift && Leading == 32 && Trailing > 0 &&
            (Trailing + C1) == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Q1::SRLIW, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing, DL, VT)),
                        0);
          return true;
        }
        // Look for (srl (and X, Mask), C1) where Mask has 32 leading zeros and
        // C3 trailing zeros. If C3-C1==ShAmt we can use SRLIW+SHXADD.
        if (!LeftShift && Leading == 32 && Trailing > C1 &&
            (Trailing - C1) == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Q1::SRLIW, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing, DL, VT)),
                        0);
          return true;
        }
      }
    }
  }

  return false;
}

/// Look for various patterns that can be done with a SHL that can be folded
/// into a SHXADD_UW. \p ShAmt contains 1, 2, or 3 and is set based on which
/// SHXADD_UW we are trying to match.
bool Q1DAGToDAGISel::selectSHXADD_UWOp(SDValue N, unsigned ShAmt,
                                          SDValue &Val) {
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1)) &&
      N.hasOneUse()) {
    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.hasOneUse()) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      Mask &= maskTrailingZeros<uint64_t>(C2);

      // Look for (and (shl y, c2), c1) where c1 is a shifted mask with
      // 32-ShAmt leading zeros and c2 trailing zeros. We can use SLLI by
      // c2-ShAmt followed by SHXADD_UW with ShAmt for the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned Leading = llvm::countl_zero(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        if (Leading == 32 - ShAmt && Trailing == C2 && Trailing > ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Q1::SLLI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(C2 - ShAmt, DL, VT)),
                        0);
          return true;
        }
      }
    }
  }

  return false;
}

static bool vectorPseudoHasAllNBitUsers(SDNode *User, unsigned UserOpNo,
                                        unsigned Bits,
                                        const TargetInstrInfo *TII) {
  unsigned MCOpcode = Q1::getRVVMCOpcode(User->getMachineOpcode());

  if (!MCOpcode)
    return false;

  const MCInstrDesc &MCID = TII->get(User->getMachineOpcode());
  const uint64_t TSFlags = MCID.TSFlags;
  if (!Q1II::hasSEWOp(TSFlags))
    return false;
  assert(Q1II::hasVLOp(TSFlags));

  bool HasGlueOp = User->getGluedNode() != nullptr;
  unsigned ChainOpIdx = User->getNumOperands() - HasGlueOp - 1;
  bool HasChainOp = User->getOperand(ChainOpIdx).getValueType() == MVT::Other;
  bool HasVecPolicyOp = Q1II::hasVecPolicyOp(TSFlags);
  unsigned VLIdx =
      User->getNumOperands() - HasVecPolicyOp - HasChainOp - HasGlueOp - 2;
  const unsigned Log2SEW = User->getConstantOperandVal(VLIdx + 1);

  if (UserOpNo == VLIdx)
    return false;

  auto NumDemandedBits =
      Q1::getVectorLowDemandedScalarBits(MCOpcode, Log2SEW);
  return NumDemandedBits && Bits >= *NumDemandedBits;
}

// Return true if all users of this SDNode* only consume the lower \p Bits.
// This can be used to form W instructions for add/sub/mul/shl even when the
// root isn't a sext_inreg. This can allow the ADDW/SUBW/MULW/SLLIW to CSE if
// SimplifyDemandedBits has made it so some users see a sext_inreg and some
// don't. The sext_inreg+add/sub/mul/shl will get selected, but still leave
// the add/sub/mul/shl to become non-W instructions. By checking the users we
// may be able to use a W instruction and CSE with the other instruction if
// this has happened. We could try to detect that the CSE opportunity exists
// before doing this, but that would be more complicated.
bool Q1DAGToDAGISel::hasAllNBitUsers(SDNode *Node, unsigned Bits,
                                        const unsigned Depth) const {
  assert((Node->getOpcode() == ISD::ADD || Node->getOpcode() == ISD::SUB ||
          Node->getOpcode() == ISD::MUL || Node->getOpcode() == ISD::SHL ||
          Node->getOpcode() == ISD::SRL || Node->getOpcode() == ISD::AND ||
          Node->getOpcode() == ISD::OR || Node->getOpcode() == ISD::XOR ||
          Node->getOpcode() == ISD::SIGN_EXTEND_INREG ||
          isa<ConstantSDNode>(Node) || Depth != 0) &&
         "Unexpected opcode");

  if (Depth >= SelectionDAG::MaxRecursionDepth)
    return false;

  // The PatFrags that call this may run before Q1GenDAGISel.inc has checked
  // the VT. Ensure the type is scalar to avoid wasting time on vectors.
  if (Depth == 0 && !Node->getValueType(0).isScalarInteger())
    return false;

  for (auto UI = Node->use_begin(), UE = Node->use_end(); UI != UE; ++UI) {
    SDNode *User = *UI;
    // Users of this node should have already been instruction selected
    if (!User->isMachineOpcode())
      return false;

    // TODO: Add more opcodes?
    switch (User->getMachineOpcode()) {
    default:
      if (vectorPseudoHasAllNBitUsers(User, UI.getOperandNo(), Bits, TII))
        break;
      return false;
    case Q1::ADDW:
    case Q1::ADDIW:
    case Q1::SUBW:
    case Q1::MULW:
    case Q1::SLLW:
    case Q1::SLLIW:
    case Q1::SRAW:
    case Q1::SRAIW:
    case Q1::SRLW:
    case Q1::SRLIW:
    case Q1::DIVW:
    case Q1::DIVUW:
    case Q1::REMW:
    case Q1::REMUW:
    case Q1::ROLW:
    case Q1::RORW:
    case Q1::RORIW:
    case Q1::CLZW:
    case Q1::CTZW:
    case Q1::CPOPW:
    case Q1::SLLI_UW:
    case Q1::FMV_W_X:
    case Q1::FCVT_H_W:
    case Q1::FCVT_H_W_INX:
    case Q1::FCVT_H_WU:
    case Q1::FCVT_H_WU_INX:
    case Q1::FCVT_S_W:
    case Q1::FCVT_S_W_INX:
    case Q1::FCVT_S_WU:
    case Q1::FCVT_S_WU_INX:
    case Q1::FCVT_D_W:
    case Q1::FCVT_D_W_INX:
    case Q1::FCVT_D_WU:
    case Q1::FCVT_D_WU_INX:
    case Q1::TH_REVW:
    case Q1::TH_SRRIW:
      if (Bits >= 32)
        break;
      return false;
    case Q1::SLL:
    case Q1::SRA:
    case Q1::SRL:
    case Q1::ROL:
    case Q1::ROR:
    case Q1::BSET:
    case Q1::BCLR:
    case Q1::BINV:
      // Shift amount operands only use log2(Xlen) bits.
      if (UI.getOperandNo() == 1 && Bits >= Log2_32(Subtarget->getXLen()))
        break;
      return false;
    case Q1::SLLI:
      // SLLI only uses the lower (XLen - ShAmt) bits.
      if (Bits >= Subtarget->getXLen() - User->getConstantOperandVal(1))
        break;
      return false;
    case Q1::ANDI:
      if (Bits >= (unsigned)llvm::bit_width(User->getConstantOperandVal(1)))
        break;
      goto RecCheck;
    case Q1::ORI: {
      uint64_t Imm = cast<ConstantSDNode>(User->getOperand(1))->getSExtValue();
      if (Bits >= (unsigned)llvm::bit_width<uint64_t>(~Imm))
        break;
      [[fallthrough]];
    }
    case Q1::AND:
    case Q1::OR:
    case Q1::XOR:
    case Q1::XORI:
    case Q1::ANDN:
    case Q1::ORN:
    case Q1::XNOR:
    case Q1::SH1ADD:
    case Q1::SH2ADD:
    case Q1::SH3ADD:
    RecCheck:
      if (hasAllNBitUsers(User, Bits, Depth + 1))
        break;
      return false;
    case Q1::SRLI: {
      unsigned ShAmt = User->getConstantOperandVal(1);
      // If we are shifting right by less than Bits, and users don't demand any
      // bits that were shifted into [Bits-1:0], then we can consider this as an
      // N-Bit user.
      if (Bits > ShAmt && hasAllNBitUsers(User, Bits - ShAmt, Depth + 1))
        break;
      return false;
    }
    case Q1::SEXT_B:
    case Q1::PACKH:
      if (Bits >= 8)
        break;
      return false;
    case Q1::SEXT_H:
    case Q1::FMV_H_X:
    case Q1::ZEXT_H_RV32:
    case Q1::ZEXT_H_RV64:
    case Q1::PACKW:
      if (Bits >= 16)
        break;
      return false;
    case Q1::PACK:
      if (Bits >= (Subtarget->getXLen() / 2))
        break;
      return false;
    case Q1::ADD_UW:
    case Q1::SH1ADD_UW:
    case Q1::SH2ADD_UW:
    case Q1::SH3ADD_UW:
      // The first operand to add.uw/shXadd.uw is implicitly zero extended from
      // 32 bits.
      if (UI.getOperandNo() == 0 && Bits >=  32)
        break;
      return false;
    case Q1::SB:
      if (UI.getOperandNo() == 0 && Bits >= 8)
        break;
      return false;
    case Q1::SH:
      if (UI.getOperandNo() == 0 && Bits >= 16)
        break;
      return false;
    case Q1::SW:
      if (UI.getOperandNo() == 0 && Bits >= 32)
        break;
      return false;
    }
  }

  return true;
}

// Select a constant that can be represented as (sign_extend(imm5) << imm2).
bool Q1DAGToDAGISel::selectSimm5Shl2(SDValue N, SDValue &Simm5,
                                        SDValue &Shl2) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    int64_t Offset = C->getSExtValue();
    int64_t Shift;
    for (Shift = 0; Shift < 4; Shift++)
      if (isInt<5>(Offset >> Shift) && ((Offset % (1LL << Shift)) == 0))
        break;

    // Constant cannot be encoded.
    if (Shift == 4)
      return false;

    EVT Ty = N->getValueType(0);
    Simm5 = CurDAG->getSignedConstant(Offset >> Shift, SDLoc(N), Ty,
                                      /*isTarget=*/true);
    Shl2 = CurDAG->getTargetConstant(Shift, SDLoc(N), Ty);
    return true;
  }

  return false;
}

// Select VL as a 5 bit immediate or a value that will become a register. This
// allows us to choose betwen VSETIVLI or VSETVLI later.
bool Q1DAGToDAGISel::selectVLOp(SDValue N, SDValue &VL) {
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (C && isUInt<5>(C->getZExtValue())) {
    VL = CurDAG->getTargetConstant(C->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
  } else if (C && C->isAllOnes()) {
    // Treat all ones as VLMax.
    VL = CurDAG->getSignedConstant(Q1::VLMaxSentinel, SDLoc(N),
                                   N->getValueType(0), /*isTarget=*/true);
  } else if (isa<RegisterSDNode>(N) &&
             cast<RegisterSDNode>(N)->getReg() == Q1::X0) {
    // All our VL operands use an operand that allows GPRNoX0 or an immediate
    // as the register class. Convert X0 to a special immediate to pass the
    // MachineVerifier. This is recognized specially by the vsetvli insertion
    // pass.
    VL = CurDAG->getSignedConstant(Q1::VLMaxSentinel, SDLoc(N),
                                   N->getValueType(0), /*isTarget=*/true);
  } else {
    VL = N;
  }

  return true;
}

static SDValue findVSplat(SDValue N) {
  if (N.getOpcode() == ISD::INSERT_SUBVECTOR) {
    if (!N.getOperand(0).isUndef())
      return SDValue();
    N = N.getOperand(1);
  }
  SDValue Splat = N;
  if ((Splat.getOpcode() != Q1ISD::VMV_V_X_VL &&
       Splat.getOpcode() != Q1ISD::VMV_S_X_VL) ||
      !Splat.getOperand(0).isUndef())
    return SDValue();
  assert(Splat.getNumOperands() == 3 && "Unexpected number of operands");
  return Splat;
}

bool Q1DAGToDAGISel::selectVSplat(SDValue N, SDValue &SplatVal) {
  SDValue Splat = findVSplat(N);
  if (!Splat)
    return false;

  SplatVal = Splat.getOperand(1);
  return true;
}

static bool selectVSplatImmHelper(SDValue N, SDValue &SplatVal,
                                  SelectionDAG &DAG,
                                  const Q1Subtarget &Subtarget,
                                  std::function<bool(int64_t)> ValidateImm) {
  SDValue Splat = findVSplat(N);
  if (!Splat || !isa<ConstantSDNode>(Splat.getOperand(1)))
    return false;

  const unsigned SplatEltSize = Splat.getScalarValueSizeInBits();
  assert(Subtarget.getXLenVT() == Splat.getOperand(1).getSimpleValueType() &&
         "Unexpected splat operand type");

  // The semantics of Q1ISD::VMV_V_X_VL is that when the operand
  // type is wider than the resulting vector element type: an implicit
  // truncation first takes place. Therefore, perform a manual
  // truncation/sign-extension in order to ignore any truncated bits and catch
  // any zero-extended immediate.
  // For example, we wish to match (i8 -1) -> (XLenVT 255) as a simm5 by first
  // sign-extending to (XLenVT -1).
  APInt SplatConst = Splat.getConstantOperandAPInt(1).sextOrTrunc(SplatEltSize);

  int64_t SplatImm = SplatConst.getSExtValue();

  if (!ValidateImm(SplatImm))
    return false;

  SplatVal = DAG.getSignedConstant(SplatImm, SDLoc(N), Subtarget.getXLenVT(),
                                   /*isTarget=*/true);
  return true;
}

bool Q1DAGToDAGISel::selectVSplatSimm5(SDValue N, SDValue &SplatVal) {
  return selectVSplatImmHelper(N, SplatVal, *CurDAG, *Subtarget,
                               [](int64_t Imm) { return isInt<5>(Imm); });
}

bool Q1DAGToDAGISel::selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal) {
  return selectVSplatImmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [](int64_t Imm) { return (isInt<5>(Imm) && Imm != -16) || Imm == 16; });
}

bool Q1DAGToDAGISel::selectVSplatSimm5Plus1NonZero(SDValue N,
                                                      SDValue &SplatVal) {
  return selectVSplatImmHelper(
      N, SplatVal, *CurDAG, *Subtarget, [](int64_t Imm) {
        return Imm != 0 && ((isInt<5>(Imm) && Imm != -16) || Imm == 16);
      });
}

bool Q1DAGToDAGISel::selectVSplatUimm(SDValue N, unsigned Bits,
                                         SDValue &SplatVal) {
  return selectVSplatImmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [Bits](int64_t Imm) { return isUIntN(Bits, Imm); });
}

bool Q1DAGToDAGISel::selectLow8BitsVSplat(SDValue N, SDValue &SplatVal) {
  auto IsExtOrTrunc = [](SDValue N) {
    switch (N->getOpcode()) {
    case ISD::SIGN_EXTEND:
    case ISD::ZERO_EXTEND:
    // There's no passthru on these _VL nodes so any VL/mask is ok, since any
    // inactive elements will be undef.
    case Q1ISD::TRUNCATE_VECTOR_VL:
    case Q1ISD::VSEXT_VL:
    case Q1ISD::VZEXT_VL:
      return true;
    default:
      return false;
    }
  };

  // We can have multiple nested nodes, so unravel them all if needed.
  while (IsExtOrTrunc(N)) {
    if (!N.hasOneUse() || N.getScalarValueSizeInBits() < 8)
      return false;
    N = N->getOperand(0);
  }

  return selectVSplat(N, SplatVal);
}

bool Q1DAGToDAGISel::selectScalarFPAsInt(SDValue N, SDValue &Imm) {
  // Allow bitcasts from XLenVT -> FP.
  if (N.getOpcode() == ISD::BITCAST &&
      N.getOperand(0).getValueType() == Subtarget->getXLenVT()) {
    Imm = N.getOperand(0);
    return true;
  }
  // Allow moves from XLenVT to FP.
  if (N.getOpcode() == Q1ISD::FMV_H_X ||
      N.getOpcode() == Q1ISD::FMV_W_X_RV64) {
    Imm = N.getOperand(0);
    return true;
  }

  // Otherwise, look for FP constants that can materialized with scalar int.
  ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N.getNode());
  if (!CFP)
    return false;
  const APFloat &APF = CFP->getValueAPF();
  // td can handle +0.0 already.
  if (APF.isPosZero())
    return false;

  MVT VT = CFP->getSimpleValueType(0);

  MVT XLenVT = Subtarget->getXLenVT();
  if (VT == MVT::f64 && !Subtarget->is64Bit()) {
    assert(APF.isNegZero() && "Unexpected constant.");
    return false;
  }
  SDLoc DL(N);
  Imm = selectImm(CurDAG, DL, XLenVT, APF.bitcastToAPInt().getSExtValue(),
                  *Subtarget);
  return true;
}

bool Q1DAGToDAGISel::selectRVVSimm5(SDValue N, unsigned Width,
                                       SDValue &Imm) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    int64_t ImmVal = SignExtend64(C->getSExtValue(), Width);

    if (!isInt<5>(ImmVal))
      return false;

    Imm = CurDAG->getSignedConstant(ImmVal, SDLoc(N), Subtarget->getXLenVT(),
                                    /*isTarget=*/true);
    return true;
  }

  return false;
}

// Try to remove sext.w if the input is a W instruction or can be made into
// a W instruction cheaply.
bool Q1DAGToDAGISel::doPeepholeSExtW(SDNode *N) {
  // Look for the sext.w pattern, addiw rd, rs1, 0.
  if (N->getMachineOpcode() != Q1::ADDIW ||
      !isNullConstant(N->getOperand(1)))
    return false;

  SDValue N0 = N->getOperand(0);
  if (!N0.isMachineOpcode())
    return false;

  switch (N0.getMachineOpcode()) {
  default:
    break;
  case Q1::ADD:
  case Q1::ADDI:
  case Q1::SUB:
  case Q1::MUL:
  case Q1::SLLI: {
    // Convert sext.w+add/sub/mul to their W instructions. This will create
    // a new independent instruction. This improves latency.
    unsigned Opc;
    switch (N0.getMachineOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case Q1::ADD:  Opc = Q1::ADDW;  break;
    case Q1::ADDI: Opc = Q1::ADDIW; break;
    case Q1::SUB:  Opc = Q1::SUBW;  break;
    case Q1::MUL:  Opc = Q1::MULW;  break;
    case Q1::SLLI: Opc = Q1::SLLIW; break;
    }

    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);

    // Shift amount needs to be uimm5.
    if (N0.getMachineOpcode() == Q1::SLLI &&
        !isUInt<5>(cast<ConstantSDNode>(N01)->getSExtValue()))
      break;

    SDNode *Result =
        CurDAG->getMachineNode(Opc, SDLoc(N), N->getValueType(0),
                               N00, N01);
    ReplaceUses(N, Result);
    return true;
  }
  case Q1::ADDW:
  case Q1::ADDIW:
  case Q1::SUBW:
  case Q1::MULW:
  case Q1::SLLIW:
  case Q1::PACKW:
  case Q1::TH_MULAW:
  case Q1::TH_MULAH:
  case Q1::TH_MULSW:
  case Q1::TH_MULSH:
    if (N0.getValueType() == MVT::i32)
      break;

    // Result is already sign extended just remove the sext.w.
    // NOTE: We only handle the nodes that are selected with hasAllWUsers.
    ReplaceUses(N, N0.getNode());
    return true;
  }

  return false;
}

// After ISel, a vector pseudo's mask will be copied to V0 via a CopyToReg
// that's glued to the pseudo. This tries to look up the value that was copied
// to V0.
static SDValue getMaskSetter(SDValue MaskOp, SDValue GlueOp) {
  // Check that we're using V0 as a mask register.
  if (!isa<RegisterSDNode>(MaskOp) ||
      cast<RegisterSDNode>(MaskOp)->getReg() != Q1::V0)
    return SDValue();

  // The glued user defines V0.
  const auto *Glued = GlueOp.getNode();

  if (!Glued || Glued->getOpcode() != ISD::CopyToReg)
    return SDValue();

  // Check that we're defining V0 as a mask register.
  if (!isa<RegisterSDNode>(Glued->getOperand(1)) ||
      cast<RegisterSDNode>(Glued->getOperand(1))->getReg() != Q1::V0)
    return SDValue();

  SDValue MaskSetter = Glued->getOperand(2);

  // Sometimes the VMSET is wrapped in a COPY_TO_REGCLASS, e.g. if the mask came
  // from an extract_subvector or insert_subvector.
  if (MaskSetter->isMachineOpcode() &&
      MaskSetter->getMachineOpcode() == Q1::COPY_TO_REGCLASS)
    MaskSetter = MaskSetter->getOperand(0);

  return MaskSetter;
}

static bool usesAllOnesMask(SDValue MaskOp, SDValue GlueOp) {
  // Check the instruction defining V0; it needs to be a VMSET pseudo.
  SDValue MaskSetter = getMaskSetter(MaskOp, GlueOp);
  if (!MaskSetter)
    return false;

  const auto IsVMSet = [](unsigned Opc) {
    return Opc == Q1::PseudoVMSET_M_B1 || Opc == Q1::PseudoVMSET_M_B16 ||
           Opc == Q1::PseudoVMSET_M_B2 || Opc == Q1::PseudoVMSET_M_B32 ||
           Opc == Q1::PseudoVMSET_M_B4 || Opc == Q1::PseudoVMSET_M_B64 ||
           Opc == Q1::PseudoVMSET_M_B8;
  };

  // TODO: Check that the VMSET is the expected bitwidth? The pseudo has
  // undefined behaviour if it's the wrong bitwidth, so we could choose to
  // assume that it's all-ones? Same applies to its VL.
  return MaskSetter->isMachineOpcode() &&
         IsVMSet(MaskSetter.getMachineOpcode());
}

// Return true if we can make sure mask of N is all-ones mask.
static bool usesAllOnesMask(SDNode *N, unsigned MaskOpIdx) {
  return usesAllOnesMask(N->getOperand(MaskOpIdx),
                         N->getOperand(N->getNumOperands() - 1));
}

static bool isImplicitDef(SDValue V) {
  if (!V.isMachineOpcode())
    return false;
  if (V.getMachineOpcode() == TargetOpcode::REG_SEQUENCE) {
    for (unsigned I = 1; I < V.getNumOperands(); I += 2)
      if (!isImplicitDef(V.getOperand(I)))
        return false;
    return true;
  }
  return V.getMachineOpcode() == TargetOpcode::IMPLICIT_DEF;
}

// Optimize masked RVV pseudo instructions with a known all-ones mask to their
// corresponding "unmasked" pseudo versions. The mask we're interested in will
// take the form of a V0 physical register operand, with a glued
// register-setting instruction.
bool Q1DAGToDAGISel::doPeepholeMaskedRVV(MachineSDNode *N) {
  const Q1::Q1MaskedPseudoInfo *I =
      Q1::getMaskedPseudoInfo(N->getMachineOpcode());
  if (!I)
    return false;

  unsigned MaskOpIdx = I->MaskOpIdx;
  if (!usesAllOnesMask(N, MaskOpIdx))
    return false;

  // There are two classes of pseudos in the table - compares and
  // everything else.  See the comment on Q1MaskedPseudo for details.
  const unsigned Opc = I->UnmaskedPseudo;
  const MCInstrDesc &MCID = TII->get(Opc);
  const bool UseTUPseudo = Q1II::hasVecPolicyOp(MCID.TSFlags);
#ifndef NDEBUG
  const MCInstrDesc &MaskedMCID = TII->get(N->getMachineOpcode());
  assert(Q1II::hasVecPolicyOp(MaskedMCID.TSFlags) ==
         Q1II::hasVecPolicyOp(MCID.TSFlags) &&
         "Masked and unmasked pseudos are inconsistent");
  const bool HasTiedDest = Q1II::isFirstDefTiedToFirstUse(MCID);
  assert(UseTUPseudo == HasTiedDest && "Unexpected pseudo structure");
#endif

  SmallVector<SDValue, 8> Ops;
  // Skip the passthru operand at index 0 if !UseTUPseudo.
  for (unsigned I = !UseTUPseudo, E = N->getNumOperands(); I != E; I++) {
    // Skip the mask, and the Glue.
    SDValue Op = N->getOperand(I);
    if (I == MaskOpIdx || Op.getValueType() == MVT::Glue)
      continue;
    Ops.push_back(Op);
  }

  // Transitively apply any node glued to our new node.
  const auto *Glued = N->getGluedNode();
  if (auto *TGlued = Glued->getGluedNode())
    Ops.push_back(SDValue(TGlued, TGlued->getNumValues() - 1));

  MachineSDNode *Result =
      CurDAG->getMachineNode(Opc, SDLoc(N), N->getVTList(), Ops);

  if (!N->memoperands_empty())
    CurDAG->setNodeMemRefs(Result, N->memoperands());

  Result->setFlags(N->getFlags());
  ReplaceUses(N, Result);

  return true;
}

static bool IsVMerge(SDNode *N) {
  return Q1::getRVVMCOpcode(N->getMachineOpcode()) == Q1::VMERGE_VVM;
}

// Try to fold away VMERGE_VVM instructions into their true operands:
//
// %true = PseudoVADD_VV ...
// %x = PseudoVMERGE_VVM %false, %false, %true, %mask
// ->
// %x = PseudoVADD_VV_MASK %false, ..., %mask
//
// We can only fold if vmerge's passthru operand, vmerge's false operand and
// %true's passthru operand (if it has one) are the same. This is because we
// have to consolidate them into one passthru operand in the result.
//
// If %true is masked, then we can use its mask instead of vmerge's if vmerge's
// mask is all ones.
//
// The resulting VL is the minimum of the two VLs.
//
// The resulting policy is the effective policy the vmerge would have had,
// i.e. whether or not it's passthru operand was implicit-def.
bool Q1DAGToDAGISel::performCombineVMergeAndVOps(SDNode *N) {
  SDValue Passthru, False, True, VL, Mask, Glue;
  assert(IsVMerge(N));
  Passthru = N->getOperand(0);
  False = N->getOperand(1);
  True = N->getOperand(2);
  Mask = N->getOperand(3);
  VL = N->getOperand(4);
  // We always have a glue node for the mask at v0.
  Glue = N->getOperand(N->getNumOperands() - 1);
  assert(cast<RegisterSDNode>(Mask)->getReg() == Q1::V0);
  assert(Glue.getValueType() == MVT::Glue);

  // If the EEW of True is different from vmerge's SEW, then we can't fold.
  if (True.getSimpleValueType() != N->getSimpleValueType(0))
    return false;

  // We require that either passthru and false are the same, or that passthru
  // is undefined.
  if (Passthru != False && !isImplicitDef(Passthru))
    return false;

  assert(True.getResNo() == 0 &&
         "Expect True is the first output of an instruction.");

  // Need N is the exactly one using True.
  if (!True.hasOneUse())
    return false;

  if (!True.isMachineOpcode())
    return false;

  unsigned TrueOpc = True.getMachineOpcode();
  const MCInstrDesc &TrueMCID = TII->get(TrueOpc);
  uint64_t TrueTSFlags = TrueMCID.TSFlags;
  bool HasTiedDest = Q1II::isFirstDefTiedToFirstUse(TrueMCID);

  const Q1::Q1MaskedPseudoInfo *Info =
      Q1::lookupMaskedIntrinsicByUnmasked(TrueOpc);
  if (!Info)
    return false;

  // If True has a passthru operand then it needs to be the same as vmerge's
  // False, since False will be used for the result's passthru operand.
  if (HasTiedDest && !isImplicitDef(True->getOperand(0))) {
    SDValue PassthruOpTrue = True->getOperand(0);
    if (False != PassthruOpTrue)
      return false;
  }

  // Skip if True has side effect.
  if (TII->get(TrueOpc).hasUnmodeledSideEffects())
    return false;

  // The last operand of a masked instruction may be glued.
  bool HasGlueOp = True->getGluedNode() != nullptr;

  // The chain operand may exist either before the glued operands or in the last
  // position.
  unsigned TrueChainOpIdx = True.getNumOperands() - HasGlueOp - 1;
  bool HasChainOp =
      True.getOperand(TrueChainOpIdx).getValueType() == MVT::Other;

  if (HasChainOp) {
    // Avoid creating cycles in the DAG. We must ensure that none of the other
    // operands depend on True through it's Chain.
    SmallVector<const SDNode *, 4> LoopWorklist;
    SmallPtrSet<const SDNode *, 16> Visited;
    LoopWorklist.push_back(False.getNode());
    LoopWorklist.push_back(Mask.getNode());
    LoopWorklist.push_back(VL.getNode());
    LoopWorklist.push_back(Glue.getNode());
    if (SDNode::hasPredecessorHelper(True.getNode(), Visited, LoopWorklist))
      return false;
  }

  // The vector policy operand may be present for masked intrinsics
  bool HasVecPolicyOp = Q1II::hasVecPolicyOp(TrueTSFlags);
  unsigned TrueVLIndex =
      True.getNumOperands() - HasVecPolicyOp - HasChainOp - HasGlueOp - 2;
  SDValue TrueVL = True.getOperand(TrueVLIndex);
  SDValue SEW = True.getOperand(TrueVLIndex + 1);

  auto GetMinVL = [](SDValue LHS, SDValue RHS) {
    if (LHS == RHS)
      return LHS;
    if (isAllOnesConstant(LHS))
      return RHS;
    if (isAllOnesConstant(RHS))
      return LHS;
    auto *CLHS = dyn_cast<ConstantSDNode>(LHS);
    auto *CRHS = dyn_cast<ConstantSDNode>(RHS);
    if (!CLHS || !CRHS)
      return SDValue();
    return CLHS->getZExtValue() <= CRHS->getZExtValue() ? LHS : RHS;
  };

  // Because N and True must have the same passthru operand (or True's operand
  // is implicit_def), the "effective" body is the minimum of their VLs.
  SDValue OrigVL = VL;
  VL = GetMinVL(TrueVL, VL);
  if (!VL)
    return false;

  // Some operations produce different elementwise results depending on the
  // active elements, like viota.m or vredsum. This transformation is illegal
  // for these if we change the active elements (i.e. mask or VL).
  const MCInstrDesc &TrueBaseMCID = TII->get(Q1::getRVVMCOpcode(TrueOpc));
  if (Q1II::elementsDependOnVL(TrueBaseMCID.TSFlags) && (TrueVL != VL))
    return false;
  if (Q1II::elementsDependOnMask(TrueBaseMCID.TSFlags) &&
      (Mask && !usesAllOnesMask(Mask, Glue)))
    return false;

  // Make sure it doesn't raise any observable fp exceptions, since changing the
  // active elements will affect how fflags is set.
  if (mayRaiseFPException(True.getNode()) && !True->getFlags().hasNoFPExcept())
    return false;

  SDLoc DL(N);

  unsigned MaskedOpc = Info->MaskedPseudo;
#ifndef NDEBUG
  const MCInstrDesc &MaskedMCID = TII->get(MaskedOpc);
  assert(Q1II::hasVecPolicyOp(MaskedMCID.TSFlags) &&
         "Expected instructions with mask have policy operand.");
  assert(MaskedMCID.getOperandConstraint(MaskedMCID.getNumDefs(),
                                         MCOI::TIED_TO) == 0 &&
         "Expected instructions with mask have a tied dest.");
#endif

  // Use a tumu policy, relaxing it to tail agnostic provided that the passthru
  // operand is undefined.
  //
  // However, if the VL became smaller than what the vmerge had originally, then
  // elements past VL that were previously in the vmerge's body will have moved
  // to the tail. In that case we always need to use tail undisturbed to
  // preserve them.
  bool MergeVLShrunk = VL != OrigVL;
  uint64_t Policy = (isImplicitDef(Passthru) && !MergeVLShrunk)
                        ? Q1II::TAIL_AGNOSTIC
                        : /*TUMU*/ 0;
  SDValue PolicyOp =
    CurDAG->getTargetConstant(Policy, DL, Subtarget->getXLenVT());


  SmallVector<SDValue, 8> Ops;
  Ops.push_back(False);

  const bool HasRoundingMode = Q1II::hasRoundModeOp(TrueTSFlags);
  const unsigned NormalOpsEnd = TrueVLIndex - HasRoundingMode;
  Ops.append(True->op_begin() + HasTiedDest, True->op_begin() + NormalOpsEnd);

  Ops.push_back(Mask);

  // For unmasked "VOp" with rounding mode operand, that is interfaces like
  // (..., rm, vl) or (..., rm, vl, policy).
  // Its masked version is (..., vm, rm, vl, policy).
  // Check the rounding mode pseudo nodes under Q1InstrInfoVPseudos.td
  if (HasRoundingMode)
    Ops.push_back(True->getOperand(TrueVLIndex - 1));

  Ops.append({VL, SEW, PolicyOp});

  // Result node should have chain operand of True.
  if (HasChainOp)
    Ops.push_back(True.getOperand(TrueChainOpIdx));

  // Add the glue for the CopyToReg of mask->v0.
  Ops.push_back(Glue);

  MachineSDNode *Result =
      CurDAG->getMachineNode(MaskedOpc, DL, True->getVTList(), Ops);
  Result->setFlags(True->getFlags());

  if (!cast<MachineSDNode>(True)->memoperands_empty())
    CurDAG->setNodeMemRefs(Result, cast<MachineSDNode>(True)->memoperands());

  // Replace vmerge.vvm node by Result.
  ReplaceUses(SDValue(N, 0), SDValue(Result, 0));

  // Replace another value of True. E.g. chain and VL.
  for (unsigned Idx = 1; Idx < True->getNumValues(); ++Idx)
    ReplaceUses(True.getValue(Idx), SDValue(Result, Idx));

  return true;
}

bool Q1DAGToDAGISel::doPeepholeMergeVVMFold() {
  bool MadeChange = false;
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    if (IsVMerge(N))
      MadeChange |= performCombineVMergeAndVOps(N);
  }
  return MadeChange;
}

/// If our passthru is an implicit_def, use noreg instead.  This side
/// steps issues with MachineCSE not being able to CSE expressions with
/// IMPLICIT_DEF operands while preserving the semantic intent. See
/// pr64282 for context. Note that this transform is the last one
/// performed at ISEL DAG to DAG.
bool Q1DAGToDAGISel::doPeepholeNoRegPassThru() {
  bool MadeChange = false;
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    const unsigned Opc = N->getMachineOpcode();
    if (!Q1VPseudosTable::getPseudoInfo(Opc) ||
        !Q1II::isFirstDefTiedToFirstUse(TII->get(Opc)) ||
        !isImplicitDef(N->getOperand(0)))
      continue;

    SmallVector<SDValue> Ops;
    Ops.push_back(CurDAG->getRegister(Q1::NoRegister, N->getValueType(0)));
    for (unsigned I = 1, E = N->getNumOperands(); I != E; I++) {
      SDValue Op = N->getOperand(I);
      Ops.push_back(Op);
    }

    MachineSDNode *Result =
      CurDAG->getMachineNode(Opc, SDLoc(N), N->getVTList(), Ops);
    Result->setFlags(N->getFlags());
    CurDAG->setNodeMemRefs(Result, cast<MachineSDNode>(N)->memoperands());
    ReplaceUses(N, Result);
    MadeChange = true;
  }
  return MadeChange;
}


// This pass converts a legalized DAG into a Q1-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createQ1ISelDag(Q1TargetMachine &TM,
                                       CodeGenOptLevel OptLevel) {
  return new Q1DAGToDAGISelLegacy(TM, OptLevel);
}

char Q1DAGToDAGISelLegacy::ID = 0;

Q1DAGToDAGISelLegacy::Q1DAGToDAGISelLegacy(Q1TargetMachine &TM,
                                                 CodeGenOptLevel OptLevel)
    : SelectionDAGISelLegacy(
          ID, std::make_unique<Q1DAGToDAGISel>(TM, OptLevel)) {}

INITIALIZE_PASS(Q1DAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)
