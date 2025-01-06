class Q1Extension<string name,
                  string fieldname,
                  string value,
                  string desc,
                  list<SubtargetFeature> impliedFeatures = [],
                  int major = 0,
                  int minor = 0>
  : SubtargetFeature<name, fieldname, value, desc, impliedFeatures> {

  // MajorVersion - The major version for this extension.
  int MajorVersion = major;

  // MinorVersion - The minor version for this extension.
  int MinorVersion = minor;
}


//===----------------------------------------------------------------------===//
// Q1 subtarget Classical features
//===----------------------------------------------------------------------===//

def FeatureCL : Q1Extension<"cl", "HasCL", "true", "Specifies Classical instructions">;


//===----------------------------------------------------------------------===//
// Q1 subtarget Quantum/RF features
//===----------------------------------------------------------------------===//

def FeatureRF : Q1Extension<"rf", "HasRF", "true", "Specifies RF up-conversion capabilities">;

def FeatureQC : Q1Extension<"qc", "HasQC", "false", "Specifies Qubit Control capabilities as in Qblox QCM">;

def FeatureQR : Q1Extension<"qr", "HasQR", "false", "Specifies Qubit Readout capabilities as in Qblox QRM">;


//===----------------------------------------------------------------------===//
// LLVM specific features and extensions
//===----------------------------------------------------------------------===//

defvar Q1HwMode = DefaultMode;

def Q1VT : ValueTypeByHwMode<[Q1HwMode], [i32]>;
