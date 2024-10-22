#ifndef LLVM_LIB_TARGET_Q1_Q1ISELDAGTODAG_H
#define LLVM_LIB_TARGET_Q1_Q1ISELDAGTODAG_H

#include "Q1.h"
#include "Q1TargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Support/KnownBits.h"

// RISC-V specific code to select RISC-V machine instructions for
// SelectionDAG operations.
namespace llvm {
class Q1DAGToDAGISel : public SelectionDAGISel {
  const Q1Subtarget *Subtarget = nullptr;

public:
  Q1DAGToDAGISel() = delete;

  explicit Q1DAGToDAGISel(Q1TargetMachine &TargetMachine,
                             CodeGenOptLevel OptLevel)
      : SelectionDAGISel(TargetMachine, OptLevel) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<Q1Subtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  void PreprocessISelDAG() override;
  void PostprocessISelDAG() override;

  void Select(SDNode *Node) override;

  bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                    InlineAsm::ConstraintCode ConstraintID,
                                    std::vector<SDValue> &OutOps) override;

  bool SelectAddrFrameIndex(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectFrameAddrRegImm(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectAddrRegImm(SDValue Addr, SDValue &Base, SDValue &Offset,
                        bool IsRV32Zdinx = false);
  bool SelectAddrRegImmRV32Zdinx(SDValue Addr, SDValue &Base, SDValue &Offset) {
    return SelectAddrRegImm(Addr, Base, Offset, true);
  }
  bool SelectAddrRegImmLsb00000(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool SelectAddrRegRegScale(SDValue Addr, unsigned MaxShiftAmount,
                             SDValue &Base, SDValue &Index, SDValue &Scale);

  template <unsigned MaxShift>
  bool SelectAddrRegRegScale(SDValue Addr, SDValue &Base, SDValue &Index,
                             SDValue &Scale) {
    return SelectAddrRegRegScale(Addr, MaxShift, Base, Index, Scale);
  }

  template <unsigned MaxShift, unsigned Bits>
  bool SelectAddrRegZextRegScale(SDValue Addr, SDValue &Base, SDValue &Index,
                                 SDValue &Scale) {
    if (SelectAddrRegRegScale(Addr, MaxShift, Base, Index, Scale)) {
      if (Index.getOpcode() == ISD::AND) {
        auto *C = dyn_cast<ConstantSDNode>(Index.getOperand(1));
        if (C && C->getZExtValue() == maskTrailingOnes<uint64_t>(Bits)) {
          Index = Index.getOperand(0);
          return true;
        }
      }
    }
    return false;
  }

  bool SelectAddrRegReg(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool tryShrinkShlLogicImm(SDNode *Node);
  bool trySignedBitfieldExtract(SDNode *Node);
  bool tryIndexedLoad(SDNode *Node);

  bool selectShiftMask(SDValue N, unsigned ShiftWidth, SDValue &ShAmt);
  bool selectShiftMaskXLen(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, Subtarget->getXLen(), ShAmt);
  }
  bool selectShiftMask32(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, 32, ShAmt);
  }

  bool selectSETCC(SDValue N, ISD::CondCode ExpectedCCVal, SDValue &Val);
  bool selectSETNE(SDValue N, SDValue &Val) {
    return selectSETCC(N, ISD::SETNE, Val);
  }
  bool selectSETEQ(SDValue N, SDValue &Val) {
    return selectSETCC(N, ISD::SETEQ, Val);
  }

  bool selectSExtBits(SDValue N, unsigned Bits, SDValue &Val);
  template <unsigned Bits> bool selectSExtBits(SDValue N, SDValue &Val) {
    return selectSExtBits(N, Bits, Val);
  }
  bool selectZExtBits(SDValue N, unsigned Bits, SDValue &Val);
  template <unsigned Bits> bool selectZExtBits(SDValue N, SDValue &Val) {
    return selectZExtBits(N, Bits, Val);
  }

  bool selectSHXADDOp(SDValue N, unsigned ShAmt, SDValue &Val);
  template <unsigned ShAmt> bool selectSHXADDOp(SDValue N, SDValue &Val) {
    return selectSHXADDOp(N, ShAmt, Val);
  }

  bool selectSHXADD_UWOp(SDValue N, unsigned ShAmt, SDValue &Val);
  template <unsigned ShAmt> bool selectSHXADD_UWOp(SDValue N, SDValue &Val) {
    return selectSHXADD_UWOp(N, ShAmt, Val);
  }

  bool hasAllNBitUsers(SDNode *Node, unsigned Bits,
                       const unsigned Depth = 0) const;
  bool hasAllBUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 8); }
  bool hasAllHUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 16); }
  bool hasAllWUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 32); }

  bool selectSimm5Shl2(SDValue N, SDValue &Simm5, SDValue &Shl2);

  bool selectVLOp(SDValue N, SDValue &VL);

  bool selectVSplat(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5(SDValue N, SDValue &SplatVal);
  bool selectVSplatUimm(SDValue N, unsigned Bits, SDValue &SplatVal);
  template <unsigned Bits> bool selectVSplatUimmBits(SDValue N, SDValue &Val) {
    return selectVSplatUimm(N, Bits, Val);
  }
  bool selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5Plus1NonZero(SDValue N, SDValue &SplatVal);
  // Matches the splat of a value which can be extended or truncated, such that
  // only the bottom 8 bits are preserved.
  bool selectLow8BitsVSplat(SDValue N, SDValue &SplatVal);
  bool selectScalarFPAsInt(SDValue N, SDValue &Imm);

  bool selectRVVSimm5(SDValue N, unsigned Width, SDValue &Imm);
  template <unsigned Width> bool selectRVVSimm5(SDValue N, SDValue &Imm) {
    return selectRVVSimm5(N, Width, Imm);
  }

  void addVectorLoadStoreOperands(SDNode *Node, unsigned SEWImm,
                                  const SDLoc &DL, unsigned CurOp,
                                  bool IsMasked, bool IsStridedOrIndexed,
                                  SmallVectorImpl<SDValue> &Operands,
                                  bool IsLoad = false, MVT *IndexVT = nullptr);

  void selectVLSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsStrided);
  void selectVLSEGFF(SDNode *Node, unsigned NF, bool IsMasked);
  void selectVLXSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsOrdered);
  void selectVSSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsStrided);
  void selectVSXSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsOrdered);

  void selectVSETVLI(SDNode *Node);

  void selectSF_VC_X_SE(SDNode *Node);

  // Return the RISC-V condition code that matches the given DAG integer
  // condition code. The CondCode must be one of those supported by the RISC-V
  // ISA (see translateSetCCForBranch).
  static Q1CC::CondCode getQ1CCForIntCC(ISD::CondCode CC) {
    switch (CC) {
    default:
      llvm_unreachable("Unsupported CondCode");
    case ISD::SETEQ:
      return Q1CC::COND_EQ;
    case ISD::SETNE:
      return Q1CC::COND_NE;
    case ISD::SETLT:
      return Q1CC::COND_LT;
    case ISD::SETGE:
      return Q1CC::COND_GE;
    case ISD::SETULT:
      return Q1CC::COND_LTU;
    case ISD::SETUGE:
      return Q1CC::COND_GEU;
    }
  }

// Include the pieces autogenerated from the target description.
#include "Q1GenDAGISel.inc"

private:
  bool doPeepholeSExtW(SDNode *Node);
  bool doPeepholeMaskedRVV(MachineSDNode *Node);
  bool doPeepholeMergeVVMFold();
  bool doPeepholeNoRegPassThru();
  bool performCombineVMergeAndVOps(SDNode *N);
};

class Q1DAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  explicit Q1DAGToDAGISelLegacy(Q1TargetMachine &TargetMachine,
                                   CodeGenOptLevel OptLevel);
};

namespace Q1 {
struct VLSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VSSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VLEPseudo {
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSEPseudo {
  uint16_t Masked :1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLX_VSXPseudo {
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

#define GET_Q1VSSEGTable_DECL
#define GET_Q1VLSEGTable_DECL
#define GET_Q1VLXSEGTable_DECL
#define GET_Q1VSXSEGTable_DECL
#define GET_Q1VLETable_DECL
#define GET_Q1VSETable_DECL
#define GET_Q1VLXTable_DECL
#define GET_Q1VSXTable_DECL
} // namespace Q1

} // namespace llvm

#endif
