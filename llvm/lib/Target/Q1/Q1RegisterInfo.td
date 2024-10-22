let Namespace = "Q1" in {
  class Q1Reg<bits<5> Enc, string n> : Register<n> {
    let HWEncoding = Enc;
  }

  foreach i = 0-63 in {
    def R#i : Q1Reg<i, "R"#i>;
  }

  class Q1RegisterClass<list<ValueType> regTypes, int alignment, dag regList> : RegisterClass<"Q1", regTypes, alignment, regList>;

  def GRRegs : Q1RegisterClass<[i32], 32, (add R0, R1, R2, R3, R4, R5, R6, R7, R8, R9)>;
}
