#include "Q1ISelLowering.h"
#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "Q1Subtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "Q1-lower"

// If I is a shifted mask, set the size (Width) and the
// first bit of the mask (Offset), and return true. For
// example, if I is 0x003e, (Width, Offset) = (5, 1).
static bool isShiftedMask(uint64_t I, uint64_t &Width, uint64_t &Offset) {
  if (!isShiftedMask_64(I))
    return false;

  Width = llvm::popcount(I);
  Offset = llvm::countr_zero(I);
  return true;
}

Q1TargetLowering::Q1TargetLowering(const TargetMachine &TM,
                                   const Q1Subtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  addRegisterClass(MVT::i32, &Q1::GPRegClass);

  // Compute derived properties from the register classes.
  computeRegisterProperties(Subtarget.getRegisterInfo());

  // Set up special registers.
  // setStackPointerRegisterToSaveRestore(Q1::R31);

  setBooleanContents(ZeroOrOneBooleanContent);

  const Align FunctionAlignment(4);
  setMinFunctionAlignment(FunctionAlignment);
  setPrefFunctionAlignment(FunctionAlignment);

  // TODO: add all necessary setOperationAction calls.
  setOperationAction(ISD::ADD, MVT::i32, Legal);
  setOperationAction(ISD::SUB, MVT::i32, Legal);
  setOperationAction(ISD::AND, MVT::i32, Legal);
  setOperationAction(ISD::OR, MVT::i32, Legal);
  setOperationAction(ISD::XOR, MVT::i32, Legal);
  setOperationAction(ISD::SHL, MVT::i32, Legal);

  // Special DAG combiner for bit-field operations.
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::SHL);
}

SDValue Q1TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  // TODO Implement for ops not covered by patterns in .td files
  return SDValue();
}

namespace {
SDValue performANDCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  SDValue FirstOperand = N->getOperand(0);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  // Second operand of and must be a constant.
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!Mask)
    return SDValue();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);

  SDValue NewOperand;
  unsigned Opc;

  uint64_t Offset;
  uint64_t MaskWidth, MaskOffset;
  if (isShiftedMask(Mask->getZExtValue(), MaskWidth, MaskOffset)) {
    if (FirstOperandOpc == ISD::SRL || FirstOperandOpc == ISD::SRA) {
      // Pattern match:
      // $dst = and (srl/sra $src, offset), (2**width -
      // 1)
      // => EXTU $dst, $src, width<offset>

      // The second operand of the shift must be an
      // immediate.
      ConstantSDNode *ShiftAmt =
          dyn_cast<ConstantSDNode>(FirstOperand.getOperand(1));
      if (!(ShiftAmt))
        return SDValue();

      Offset = ShiftAmt->getZExtValue();

      // Return if the shifted mask does not start at
      // bit 0 or the sum of its width and offset
      // exceeds the word's size.
      if (MaskOffset != 0 || Offset + MaskWidth > ValTy.getSizeInBits())
        return SDValue();

      Opc = Q1ISD::EXTU;
      NewOperand = FirstOperand.getOperand(0);
    } else
      return SDValue();
  } else if (isShiftedMask(~Mask->getZExtValue() &
                               ((0x1ULL << ValTy.getSizeInBits()) - 1),
                           MaskWidth, MaskOffset)) {
    // Pattern match:
    // $dst = and $src, ~((2**width - 1) << offset)
    // => CLR $dst, $src, width<offset>
    Opc = Q1ISD::CLR;
    NewOperand = FirstOperand;
    Offset = MaskOffset;
  } else
    return SDValue();
  return DAG.getNode(Opc, DL, ValTy, NewOperand,
                     DAG.getConstant(MaskWidth << 5 | Offset, DL, MVT::i32));
}

SDValue performORCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  uint64_t Width, Offset;

  // Second operand of or must be a constant shifted
  // mask.
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!Mask || !isShiftedMask(Mask->getZExtValue(), Width, Offset))
    return SDValue();

  // Pattern match:
  // $dst = or $src, ((2**width - 1) << offset
  // => SET $dst, $src, width<offset>
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  return DAG.getNode(Q1ISD::SET, DL, ValTy, N->getOperand(0),
                     DAG.getConstant(Width << 5 | Offset, DL, MVT::i32));
}

SDValue performSHLCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  // Pattern match:
  // $dst = shl (and $src, (2**width - 1)), offset
  // => MAK $dst, $src, width<offset>
  SelectionDAG &DAG = DCI.DAG;
  SDValue FirstOperand = N->getOperand(0);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  // First operdns shl must be and, second operand must
  // a constant.
  ConstantSDNode *ShiftAmt = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!ShiftAmt || FirstOperandOpc != ISD::AND)
    return SDValue();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);

  uint64_t Offset;
  uint64_t MaskWidth, MaskOffset;
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(FirstOperand->getOperand(1));
  if (!Mask || !isShiftedMask(Mask->getZExtValue(), MaskWidth, MaskOffset))
    return SDValue();

  // The second operand of the shift must be an
  // immediate.
  Offset = ShiftAmt->getZExtValue();

  // Return if the shifted mask does not start at bit 0
  // or the sum of its width and offset exceeds the
  // word's size.
  if (MaskOffset != 0 || Offset + MaskWidth > ValTy.getSizeInBits())
    return SDValue();

  return DAG.getNode(Q1ISD::MAK, DL, ValTy, FirstOperand.getOperand(0),
                     DAG.getConstant(MaskWidth << 5 | Offset, DL, MVT::i32));
}
} // namespace

SDValue Q1TargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();
  LLVM_DEBUG(dbgs() << "In PerformDAGCombine\n");

  // TODO: Match certain and/or/shift ops to ext/mak.
  unsigned Opc = N->getOpcode();

  switch (Opc) {
  default:
    break;
  case ISD::AND:
    return performANDCombine(N, DCI);
  case ISD::OR:
    return performORCombine(N, DCI);
  case ISD::SHL:
    return performSHLCombine(N, DCI);
  }

  return SDValue();
}

//===----------------------------------------------------------------------===//
//            Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "Q1GenCallingConv.inc"

SDValue Q1TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_Q1);

  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    SDValue ArgValue;
    CCValAssign &VA = ArgLocs[I];
    EVT LocVT = VA.getLocVT();
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      const TargetRegisterClass *RC;
      switch (LocVT.getSimpleVT().SimpleTy) {
      default:
        // Integers smaller than i64 should be promoted
        // to i32.
        llvm_unreachable("Unexpected argument type");
      case MVT::i32:
        RC = &Q1::GPRegClass;
        break;
      }

      Register VReg = MRI.createVirtualRegister(RC);
      MRI.addLiveIn(VA.getLocReg(), VReg);
      ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

      // If this is an 8/16-bit value, it is really
      // passed promoted to 32 bits. Insert an
      // assert[sz]ext to capture this, then truncate to
      // the right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, DL, LocVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, DL, LocVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));

      if (VA.getLocInfo() != CCValAssign::Full)
        ArgValue = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), ArgValue);

      InVals.push_back(ArgValue);
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      llvm_unreachable("Q1 - LowerFormalArguments - "
                       "Memory argument not implemented");
    }
  }

  if (IsVarArg) {
    llvm_unreachable("Q1 - LowerFormalArguments - "
                     "VarArgs not Implemented");
  }

  return Chain;
}

SDValue
Q1TargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool IsVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              const SDLoc &DL, SelectionDAG &DAG) const {
  // Assign locations to each returned value.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RetLocs,
                    *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_Q1);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];

    // Make the return register live on exit.
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Chain and glue the copies together.
    Register Reg = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, OutVals[I], Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(Q1ISD::RET_GLUE, DL, MVT::Other, RetOps);
}

SDValue Q1TargetLowering::LowerCall(CallLoweringInfo &CLI,
                                    SmallVectorImpl<SDValue> &InVals) const {
  llvm_unreachable("Q1 - LowerCall - Not Implemented");
}

const char *Q1TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
#define OPCODE(Opc)                                                            \
  case Opc:                                                                    \
    return #Opc
    OPCODE(Q1ISD::RET_GLUE);
    OPCODE(Q1ISD::CALL);
    OPCODE(Q1ISD::CLR);
    OPCODE(Q1ISD::SET);
    OPCODE(Q1ISD::EXT);
    OPCODE(Q1ISD::EXTU);
    OPCODE(Q1ISD::MAK);
    OPCODE(Q1ISD::ROT);
    OPCODE(Q1ISD::FF1);
    OPCODE(Q1ISD::FF0);
#undef OPCODE
  default:
    return nullptr;
  }
}
