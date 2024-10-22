#include "MCTargetDesc/Q1InstPrinter.h"
#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "TargetInfo/Q1TargetInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmMacro.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>

using namespace llvm;

namespace {
struct Q1Operand;

class Q1AsmParser : public MCTargetAsmParser {
  SMLoc getLoc() const { return getParser().getTok().getLoc(); }

  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  MCRegister matchRegisterNameHelper(StringRef Name) const;
  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  ParseStatus parseDirective(AsmToken DirectiveID) override;

  // Helper to actually emit an instruction to the MCStreamer. Also, when
  // possible, compression of the instruction is performed.
  void emitToStreamer(MCStreamer &S, const MCInst &Inst);

  // Check instruction constraints.
  bool validateInstruction(MCInst &Inst, OperandVector &Operands);

  /// Helper for processing MC instructions that have been successfully matched
  /// by matchAndEmitInstruction. Modifications to the emitted instructions,
  /// like the expansion of pseudo instructions (e.g., "li"), can be performed
  /// in this method.
  bool processInstruction(MCInst &Inst, SMLoc IDLoc, OperandVector &Operands,
                          MCStreamer &Out);

// Auto-generated instruction matching functions
#define GET_ASSEMBLER_HEADER
#include "Q1GenAsmMatcher.inc"

  ParseStatus parseImmediate(OperandVector &Operands);
  ParseStatus parseRegister(OperandVector &Operands);

  bool parseOperand(OperandVector &Operands, StringRef Mnemonic);

  bool parseDirectiveDEF();

public:
  Q1AsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
              const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
};

/// Q1Operand - Instances of this class represent a parsed machine
/// instruction
struct Q1Operand final : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Register,
    Immediate,
  } Kind;
  SMLoc StartLoc, EndLoc;

  struct RegOp {
    MCRegister RegNum;
    bool IsGP;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  union {
    StringRef Tok;
    RegOp Reg;
    ImmOp Imm;
  };

  Q1Operand(KindTy K) : Kind(K) {}

public:
  Q1Operand(const Q1Operand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case KindTy::Token:
      Tok = o.Tok;
      break;
    case KindTy::Register:
      Reg = o.Reg;
      break;
    case KindTy::Immediate:
      Imm = o.Imm;
      break;
    }
  }

  /// getStartLoc - Gets location of the first token of this operand
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Gets location of the last token of this operand
  SMLoc getEndLoc() const override { return EndLoc; }

  bool isToken() const override { return Kind == KindTy::Token; }
  bool isReg() const override { return Kind == KindTy::Register; }
  bool isImm() const override { return Kind == KindTy::Immediate; }
  bool isMem() const override { return false; }
  //  bool isGP() const {
  //    return Kind == KindTy::Register &&
  //           Q1MCRegisterClasses[Q1::GPRegClassID].contains(Reg.RegNum);
  //  }

  MCRegister getReg() const override {
    assert(Kind == KindTy::Register && "Invalid type access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.Val;
  }

  StringRef getToken() const {
    assert(Kind == KindTy::Token && "Invalid type access!");
    return Tok;
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case KindTy::Register:
      OS << "<register " << Q1InstPrinter::getRegisterName(getReg()) << ">";
      break;
    case KindTy::Immediate:
      OS << *getImm();
      break;
    case KindTy::Token:
      OS << "'" << getToken() << "'";
      break;
    }
  }

  static std::unique_ptr<Q1Operand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<Q1Operand>(KindTy::Token);
    Op->Tok = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<Q1Operand> createReg(MCRegister Reg, SMLoc S,
                                              SMLoc E) {
    auto Op = std::make_unique<Q1Operand>(KindTy::Register);
    Op->Reg.RegNum = Reg.id();
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<Q1Operand> createImm(const MCExpr *Val, SMLoc S,
                                              SMLoc E) {
    auto Op = std::make_unique<Q1Operand>(KindTy::Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    assert(Expr && "Expr shouldn't be null!");
    if (auto *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  // Used by the TableGen Code
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    addExpr(Inst, getImm());
  }
};
} // namespace

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "Q1GenAsmMatcher.inc"

bool Q1AsmParser::matchAndEmitInstruction(SMLoc IDLoc, unsigned int &Opcode,
                                          OperandVector &Operands,
                                          MCStreamer &Out, uint64_t &ErrorInfo,
                                          bool MatchingInlineAsm) {
  MCInst Inst;
  FeatureBitset MissingFeatures;

  auto Result = MatchInstructionImpl(Operands, Inst, ErrorInfo, MissingFeatures,
                                     MatchingInlineAsm);

  // Handle the case when the error message is of specific type
  // other than the generic Match_InvalidOperand, and the
  // corresponding operand is missing.
  if (Result > FIRST_TARGET_MATCH_RESULT_TY) {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL && ErrorInfo >= Operands.size())
      return Error(ErrorLoc, "too few operands for instruction");
  }

  switch (Result) {
  case Match_Success:
    if (validateInstruction(Inst, Operands))
      return true;
    return processInstruction(Inst, IDLoc, Operands, Out);
  case Match_MnemonicFail: {
    FeatureBitset FBS = ComputeAvailableFeatures(getSTI().getFeatureBits());
    std::string Suggestion =
        Q1MnemonicSpellCheck(((Q1Operand &)*Operands[0]).getToken(), FBS, 0);
    return Error(IDLoc, "unrecognized instruction mnemonic" + Suggestion);
  }
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(ErrorLoc, "too few operands for instruction");

      ErrorLoc = ((Q1Operand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }
  default:
    break;
  }

  llvm_unreachable("Unknown match type detected!");
}

MCRegister Q1AsmParser::matchRegisterNameHelper(StringRef Name) const {
  MCRegister Reg = MatchRegisterName(Name);
  assert(Reg <= Q1::R63 && Reg >= Q1::R0);
  // Based on the tablegen enum ordering.
  static_assert(Q1::R0 < Q1::R63, "Matching must be updated");
  if (!Reg)
    Reg = MCRegister();
  return Reg;
}

bool Q1AsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                SMLoc &EndLoc) {
  if (!tryParseRegister(Reg, StartLoc, EndLoc).isSuccess())
    return Error(StartLoc, "invalid register name");
  return false;
}

ParseStatus Q1AsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                          SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  StringRef Name = getLexer().getTok().getIdentifier();

  Reg = matchRegisterNameHelper(Name);
  if (!Reg)
    return ParseStatus::NoMatch;

  getParser().Lex(); // Eat identifier token.
  return ParseStatus::Success;
}

bool Q1AsmParser::parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                   SMLoc NameLoc, OperandVector &Operands) {
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(Q1Operand::createToken(Name, NameLoc));

  // If there are no more operands, then finish
  if (getLexer().is(AsmToken::EndOfStatement)) {
    getParser().Lex(); // Consume the EndOfStatement.
    return false;
  }

  // Parse first operand
  if (parseOperand(Operands, Name))
    return true;

  // Parse until end of statement, consuming commas between operands
  while (parseOptionalToken(AsmToken::Comma)) {
    // Parse next operand
    if (parseOperand(Operands, Name))
      return true;
  }

  if (getParser().parseEOL("unexpected token")) {
    getParser().eatToEndOfStatement();
    return true;
  }
  return false;
}

ParseStatus Q1AsmParser::parseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".DEF")
    return parseDirectiveDEF();
  return ParseStatus::NoMatch;
}

void Q1AsmParser::emitToStreamer(MCStreamer &S, const MCInst &Inst) {
  S.emitInstruction(Inst, getSTI());
}

bool Q1AsmParser::validateInstruction(MCInst &Inst, OperandVector &Operands) {
  // todo - validation
  return false;
}

bool Q1AsmParser::processInstruction(MCInst &Inst, SMLoc IDLoc,
                                     OperandVector &Operands, MCStreamer &Out) {
  emitToStreamer(Out, Inst);
  return false;
}

ParseStatus Q1AsmParser::parseImmediate(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;
  const MCExpr *Res;

  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::Integer:
    if (getParser().parseExpression(Res, E))
      return ParseStatus::Failure;
    break;
  }

  Operands.push_back(Q1Operand::createImm(Res, S, E));
  return ParseStatus::Success;
}

ParseStatus Q1AsmParser::parseRegister(OperandVector &Operands) {
  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::Identifier:
    StringRef Name = getLexer().getTok().getIdentifier();
    MCRegister Reg = matchRegisterNameHelper(Name);

    if (!Reg) {
      return ParseStatus::NoMatch;
    }
    SMLoc S = getLoc();
    SMLoc E = SMLoc::getFromPointer(S.getPointer() + Name.size());
    getLexer().Lex();
    Operands.push_back(Q1Operand::createReg(Reg, S, E));
    return ParseStatus::Success;
  }
}

bool Q1AsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  // Attempt to parse token as a register.
  if (parseRegister(Operands).isSuccess())
    return false;

  // Attempt to parse token as an immediate
  if (parseImmediate(Operands).isSuccess()) {
    return false;
  }

  // Finally we have exhausted all options and must declare defeat.
  Error(getLoc(), "unknown operand");
  return true;
}

/// parseDirectiveDEF
///  ::= .DEF name value
bool Q1AsmParser::parseDirectiveDEF() {
  MCAsmParser &Parser = getParser();

  if (!(Parser.getTok().is(AsmToken::Identifier)))
    Error(getLoc(), "unknown identifier name");

  Parser.Lex();

  if (!(Parser.getTok().is(AsmToken::String)))
    Error(getLoc(), "unknown string value");

  // If there are no more operands, then finish
  if (getLexer().is(AsmToken::EndOfStatement)) {
    Parser.Lex(); // Consume the EndOfStatement.
    return false;
  }

  return false;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeQ1AsmParser() {
  RegisterMCAsmParser<Q1AsmParser> X(getTheQ1Target());
}
