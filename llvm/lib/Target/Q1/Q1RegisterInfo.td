//===----------------------------------------------------------------------===//
// Registers
//===----------------------------------------------------------------------===//

class Q1Reg<bits<6> Enc, string n> : Register<n> {
  let Namespace = "Q1";
  let HWEncoding{15-6} = 0;
  let HWEncoding{5-0} = Enc;
}

foreach i = 0-63 in {
  def R#i : Q1Reg<i, "R"#i>;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

class Q1RegisterClass<dag regList> : RegisterClass<"Q1", [i32], 32, regList>;

def GP : Q1RegisterClass<(add (sequence "R%u", 0, 63))>;
