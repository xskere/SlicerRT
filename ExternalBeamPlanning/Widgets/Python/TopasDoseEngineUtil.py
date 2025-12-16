import os
import shutil
import tempfile
import subprocess
import vtk
import slicer
import logging
import uuid
import glob

#------------------------------------------------------------------------------
#
# TopasDoseEngineUtil
#
#------------------------------------------------------------------------------
class TopasDoseEngineUtil:
  """Static utility class for TOPAS dose engine operations"""

  #------------------------------------------------------------------------------
  @staticmethod
  def exportCTAsDicomSeries(ctData, dicomDirectory):
    """Export CT volume as DICOM series in LPS orientation to dicomDirectory"""
    volumeNode = ctData['volumeNode']
    slicer.util.setSliceViewerLayers(background=volumeNode)
    # Prepare CLI parameters
    cliModule = slicer.modules.createdicomseries
    outputDir = dicomDirectory
    os.makedirs(outputDir, exist_ok=True)
    params = {
        'inputVolume': volumeNode.GetID(),
        'dicomDirectory': outputDir,
        'dicomPrefix': 'IMG',
        'dicomNumberFormat': '%04d',
        'type': 'Short',
        'patientSex': '[unknown]',
        'studyID': 'SLICER10001',
        'modality': 'CT',
        'manufacturer': 'Unknown manufacturer',
        'model': 'Unknown model',
        'seriesNumber': 301,
        'seriesDescription': 'No series description',
        'patientPosition': 'HFS',
        'windowCenter': 40.0,
        'windowWidth': 400.0,
        'rescaleIntercept': 0.0,
        'rescaleSlope': 1.0,
        'studyInstanceUID': str(uuid.uuid4()),
        'seriesInstanceUID': str(uuid.uuid4()),
        'frameOfReferenceUID': str(uuid.uuid4()),
    }
    # Run CLI synchronously
    cliNode = slicer.cli.runSync(cliModule, None, params)
    if cliNode.GetStatusString() != 'Completed':
        raise RuntimeError(f"CreateDICOMSeries CLI failed: {cliNode.GetStatusString()} - {cliNode.GetErrorText()}")
    # Check for DICOM files
    dicomFiles = glob.glob(os.path.join(outputDir, '**', '*.dcm'), recursive=True)
    if not dicomFiles:
        details = f"\nExport directory exists: {os.path.exists(outputDir)}\n"
        details += f"Export directory writable: {os.access(outputDir, os.W_OK)}\n"
        details += f"volumeNode: {volumeNode.GetName()} (ID: {volumeNode.GetID()})\n"
        raise RuntimeError(f"Failed to export CT as DICOM to {outputDir}.{details}")
    logging.info(f"Successfully exported CT as DICOM to {outputDir}, found {len(dicomFiles)} DICOM files.")
    return outputDir

  #------------------------------------------------------------------------------
  @staticmethod
  def extractCTData(planNode):
    """Extract CT volume data and properties"""
    # Get CT volume from plan
    ctVolumeNode = planNode.GetReferenceVolumeNode()
    if not ctVolumeNode:
      raise RuntimeError("No CT volume found in plan")

    # Get image data
    imageData = ctVolumeNode.GetImageData()
    if not imageData:
      raise RuntimeError("No image data in CT volume")

    # Get spacing and origin
    spacing = ctVolumeNode.GetSpacing()
    origin = ctVolumeNode.GetOrigin()
    dimensions = imageData.GetDimensions()

    # Convert to numpy array
    pointData = imageData.GetPointData()
    arrayData = pointData.GetScalars()
    numpyArray = vtk.util.numpy_support.vtk_to_numpy(arrayData)
    numpyArray = numpyArray.reshape(dimensions[2], dimensions[1], dimensions[0])

    return {
      'data': numpyArray,
      'spacing': spacing,
      'origin': origin,
      'dimensions': dimensions,
      'volumeNode': ctVolumeNode
    }

  #------------------------------------------------------------------------------
  @staticmethod
  def extractBeamProperties(beamNode):
    """Extract beam properties from beam node"""
    beamProperties = {}

    # Get beam geometry
    isocenter = [0.0, 0.0, 0.0]
    beamNode.GetPlanIsocenterPosition(isocenter)
    beamProperties['isocenter'] = isocenter

    beamProperties['gantryAngle'] = beamNode.GetGantryAngle()
    beamProperties['couchAngle'] = beamNode.GetCouchAngle()
    beamProperties['collimatorAngle'] = beamNode.GetCollimatorAngle()

    # Get beam parameters
    beamProperties['energy'] = beamNode.GetBeamEnergy()
    beamProperties['sourceAxisDistance'] = beamNode.GetSAD()

    # Get field size from jaws
    x1 = beamNode.GetX1Jaw()
    x2 = beamNode.GetX2Jaw()
    y1 = beamNode.GetY1Jaw()
    y2 = beamNode.GetY2Jaw()
    logging.info(f"Jaw positions: X1={x1}, X2={x2}, Y1={y1}, Y2={y2}")
    beamProperties['fieldSizeX'] = abs(x2 - x1)
    logging.info(f"Calculated field size X: {beamProperties['fieldSizeX']} mm")
    beamProperties['fieldSizeY'] = abs(y2 - y1)
    logging.info(f"Calculated field size Y: {beamProperties['fieldSizeY']} mm")

    # Default number of histories
    beamProperties['numberOfHistories'] = 1000000

    return beamProperties

  #------------------------------------------------------------------------------
  @staticmethod
  def createTopasInputFileDicom(dicomDirectory, beamProperties, workingDirectory, topasDirectory, ctData=None, planFilePath=None, doseFilePath=None):
    """Create TOPAS input file for dose calculation using DICOM RT Ion interface.

    Args:
      dicomDirectory: Path to directory containing CT DICOM series
      beamProperties: Dictionary containing beam parameters (from extractBeamProperties)
      workingDirectory: Working directory for TOPAS simulation
      topasDirectory: TOPAS installation directory
      planFilePath: Path to RT Ion Plan DICOM file (optional)
      doseFilePath: Path for dose output (optional, will use default if not specified)
    """
    inputFilePath = os.path.join(workingDirectory, 'topas_input.txt')

    # Set default dose output path if not provided
    if not doseFilePath:
      doseFilePath = os.path.join(workingDirectory, 'dose_output')

    # topasDirectory is already a Linux path (e.g., /home/user/topas)
    # Use forward slashes directly since this path goes into the TOPAS input file for WSL
    includeFilePathTopas = f"{topasDirectory}/examples/Patient/HUtoMaterialSchneider.txt"

    # Convert Windows paths to WSL format if on Windows (TOPAS runs in WSL)
    if TopasDoseEngineUtil.isWindows():
      dicomDirectoryTopas = TopasDoseEngineUtil.winToWslPath(dicomDirectory)
      doseFilePathTopas = TopasDoseEngineUtil.winToWslPath(doseFilePath)
      planFilePathTopas = TopasDoseEngineUtil.winToWslPath(planFilePath) if planFilePath else None
    else:
      dicomDirectoryTopas = dicomDirectory
      doseFilePathTopas = doseFilePath
      planFilePathTopas = planFilePath

    # Extract beam properties
    gantryAngle = beamProperties.get('gantryAngle', 0.0)
    couchAngle = beamProperties.get('couchAngle', 0.0)
    collimatorAngle = beamProperties.get('collimatorAngle', 0.0)
    isocenter = beamProperties.get('isocenter', [0.0, 0.0, 0.0])
    energy = beamProperties.get('energy', 100.0)  # MeV for protons
    sad = beamProperties.get('sourceAxisDistance', 1000.0)  # mm
    numberOfHistories = beamProperties.get('numberOfHistories', 1000000)
    fieldSizeX = beamProperties.get('fieldSizeX', 100.0)  # mm
    fieldSizeY = beamProperties.get('fieldSizeY', 100.0)  # mm
    logging.info(f"Beam properties extracted: energy={energy} MeV, SAD={sad} mm, histories={numberOfHistories}, field size={fieldSizeX}x{fieldSizeY} mm")

    logging.info(f"Beam configuration: energy={energy} MeV, SAD={sad} mm, histories={numberOfHistories}")
    logging.info(f"Angles: gantry={gantryAngle}°, couch={couchAngle}°, collimator={collimatorAngle}°")
    logging.info(f"Isocenter: {isocenter}")

    with open(inputFilePath, 'w') as f:
      f.write("# TOPAS Input File - RT Ion Dose Calculation\n")
      f.write("# Generated by SlicerRT TopasDoseEngine\n\n")

      # Include HU to material conversion
      f.write(f'includeFile = {includeFilePathTopas}\n\n')

      # Physics settings
      f.write("# Physics\n")
      # TODO: Add hadronic physics for nuclear interactions (e.g. "g4h-phy_QGSP_BIC" or "g4h-phy_QGSP_BIC_HP" for full accuracy).
      # This will increase simulation time by 3-10x.
      f.write('sv:Ph/Default/Modules = 1 "g4em-standard_opt4"\n')
      if planFilePathTopas:
        # Register RTION as a layered mass geometry world so apertures and range
        # shifters defined in TsRTIonComponents physically interact with the beam.
        # Without this, IsParallel = "T" alone only affects geometry, not physics.
        f.write('sv:Ph/Default/LayeredMassGeometryWorlds = 1 "RTION"\n')
      f.write('\n')

      # World geometry
      f.write("# World\n")
      f.write('s:Ge/World/Type     = "TsBox"\n')
      f.write('s:Ge/World/Material = "G4_AIR"\n')
      f.write('d:Ge/World/HLX      = 2.0 m\n')
      f.write('d:Ge/World/HLY      = 2.0 m\n')
      f.write('d:Ge/World/HLZ      = 2.0 m\n\n')

      if planFilePathTopas:
        # === DICOM RT Ion Plan path (TsRTIonSource + TsRTIonComponents) ===
        # Based on the dicom-interface tutorial (beam.txt / plan.txt):
        #   - IEC_F is a fixed-frame Group at World origin
        #   - TsDicomPatient and TsRTIonComponents are both parented to IEC_F
        #   - TsRTIonSource fires from IEC_F (not from the RTION geometry)
        #   - TsRTIonComponents reads imgdirectory to auto-compute ImgCenterX/Y/Z
        #     from the CT DICOM, so we only need dc: placeholder declarations
        #   - Both Ge/RTION and So/RTION need the same four rotations

        # IEC Fixed coordinate frame (parent for patient, beam geometry, and source)
        f.write("# IEC Fixed coordinate frame\n")
        f.write('s:Ge/IEC_F/Type   = "Group"\n')
        f.write('s:Ge/IEC_F/Parent = "World"\n')
        f.write('d:Ge/IEC_F/TransX = 0. mm\n')
        f.write('d:Ge/IEC_F/TransY = 0. mm\n')
        f.write('d:Ge/IEC_F/TransZ = 0. mm\n\n')

        # Patient from DICOM CT — centered at IEC_F origin (TsDicomPatient
        # places the CT bounding-box center at Trans=0 of its parent).
        f.write("# Patient from DICOM CT\n")
        f.write('s:Ge/Patient/Type              = "TsDicomPatient"\n')
        f.write('s:Ge/Patient/Parent            = "IEC_F"\n')
        f.write(f's:Ge/Patient/DicomDirectory    = "{dicomDirectoryTopas}"\n')
        f.write('sv:Ge/Patient/DicomModalityTags = 1 "CT"\n')
        f.write('s:Ge/Patient/ImagingtoMaterialConverter = "Schneider"\n')
        f.write('d:Ge/Patient/TransX = 0. mm\n')
        f.write('d:Ge/Patient/TransY = 0. mm\n')
        f.write('d:Ge/Patient/TransZ = 0. mm\n')
        f.write('d:Ge/Patient/RotX   = 0. deg\n')
        f.write('d:Ge/Patient/RotY   = 0. deg\n')
        f.write('d:Ge/Patient/RotZ   = 0. deg\n')

        f.write('b:Ge/Patient/IgnoreInconsistentFrameOfReferenceUID = "True"\n\n')

        # Beam-limiting device geometry from DICOM RT Ion Plan.
        # Type is "TsRTIonComponents" (plural — matches the C++ class name).
        # imgdirectory triggers automatic ImgCenter computation from the CT DICOM.
        # dc: placeholders are required before the formula lines so TOPAS can
        # parse them; TsRTIonComponents overwrites them at runtime.
        # IsParallel = "T" makes this a parallel world (doesn't interfere with
        # dose scoring on the patient).
        f.write("# Beam-limiting device geometry (DICOM RT Ion)\n")
        f.write('s:Ge/RTION/Type          = "TsRTIonComponents"\n')
        f.write('s:Ge/RTION/Parent        = "IEC_F"\n')
        f.write(f's:Ge/RTION/File          = "{planFilePathTopas}"\n')
        f.write(f's:Ge/RTION/imgdirectory  = "{dicomDirectoryTopas}"\n')
        f.write('i:Ge/RTION/BeamNumber    = 1\n')
        f.write('b:Ge/RTION/IsParallel    = "T"\n')
        # Changeable parameter placeholders (overwritten by TsRTIonComponents at runtime)
        f.write('dc:Ge/RTION/ImgCenterX       = 0 mm\n')
        f.write('dc:Ge/RTION/ImgCenterY       = 0 mm\n')
        f.write('dc:Ge/RTION/ImgCenterZ       = 0 mm\n')
        f.write('dc:Ge/RTION/IsoCenterX       = 0 mm\n')
        f.write('dc:Ge/RTION/IsoCenterY       = 0 mm\n')
        f.write('dc:Ge/RTION/IsoCenterZ       = 0 mm\n')
        f.write('dc:Ge/RTION/CollimatorAngle     = 0 deg\n')
        f.write('dc:Ge/RTION/GantryAngle         = 0 deg\n')
        f.write('dc:Ge/RTION/PatientSupportAngle = 0 deg\n')
        # Translation: offset from CT center to isocenter (both in DICOM patient coords)
        f.write('d:Ge/RTION/TransX = Ge/RTION/IsoCenterX - Ge/RTION/ImgCenterX mm\n')
        f.write('d:Ge/RTION/TransY = Ge/RTION/IsoCenterY - Ge/RTION/ImgCenterY mm\n')
        f.write('d:Ge/RTION/TransZ = Ge/RTION/IsoCenterZ - Ge/RTION/ImgCenterZ mm\n')
        # Rotations: IEC gantry/collimator/couch + IEC-to-DICOM frame correction
        f.write('d:Ge/RTION/RotCollimator      = Ge/RTION/CollimatorAngle deg\n')
        f.write('d:Ge/RTION/RotGantry          = Ge/RTION/GantryAngle deg\n')
        f.write('d:Ge/RTION/RotPatientSupport  = -1.0 * Ge/RTION/PatientSupportAngle deg\n')
        f.write('d:Ge/RTION/RotIEC2DICOM       = 90 deg\n\n')

        # RT Ion Beam Source — fires from IEC_F frame (same as the tutorial).
        # imgdirectory auto-computes ImgCenter for the shift formula.
        f.write("# RT Ion Beam Source (TsRTIonSource)\n")
        f.write('s:So/RTION/Type                   = "TsRTIonSource"\n')
        f.write('s:So/RTION/Component              = "IEC_F"\n')
        f.write(f's:So/RTION/File                   = "{planFilePathTopas}"\n')
        f.write(f's:So/RTION/imgdirectory           = "{dicomDirectoryTopas}"\n')
        f.write('i:So/RTION/BeamNumber             = 1\n')
        f.write(f'd:So/RTION/SID                    = {sad} mm\n')
        f.write('u:So/RTION/ParticlesPerHistory    = 1\n')
        # Changeable parameter placeholders (overwritten by TsRTIonSource at runtime)
        f.write('dc:So/RTION/ImgCenterX       = 0 mm\n')
        f.write('dc:So/RTION/ImgCenterY       = 0 mm\n')
        f.write('dc:So/RTION/ImgCenterZ       = 0 mm\n')
        f.write('dc:So/RTION/IsoCenterX       = 0 mm\n')
        f.write('dc:So/RTION/IsoCenterY       = 0 mm\n')
        f.write('dc:So/RTION/IsoCenterZ       = 0 mm\n')
        f.write('dc:So/RTION/CollimatorAngle     = 0 deg\n')
        f.write('dc:So/RTION/GantryAngle         = 0 deg\n')
        f.write('dc:So/RTION/PatientSupportAngle = 0 deg\n')
        # Shift source spots from CT center to isocenter
        f.write('d:So/RTION/ShiftX = So/RTION/IsoCenterX - So/RTION/ImgCenterX mm\n')
        f.write('d:So/RTION/ShiftY = So/RTION/IsoCenterY - So/RTION/ImgCenterY mm\n')
        f.write('d:So/RTION/ShiftZ = So/RTION/IsoCenterZ - So/RTION/ImgCenterZ mm\n')
        # Rotations: same four as on the geometry component
        f.write('d:So/RTION/RotCollimator      = So/RTION/CollimatorAngle deg\n')
        f.write('d:So/RTION/RotGantry          = So/RTION/GantryAngle deg\n')
        f.write('d:So/RTION/RotPatientSupport  = -1.0 * So/RTION/PatientSupportAngle deg\n')
        f.write('d:So/RTION/RotIEC2DICOM       = 90 deg\n')
        f.write(f'i:So/RTION/NumberOfHistoriesInRun = {numberOfHistories}\n\n')

      else:
        # === Simple beam path (no RT Ion Plan file) ===

        # Isocenter in LPS (TOPAS World) coordinates
        # Slicer RAS -> DICOM/LPS: flip X and Y
        isoCenterLpsX = -isocenter[0]
        isoCenterLpsY = -isocenter[1]
        isoCenterLpsZ =  isocenter[2]

        # Compute CT center in LPS via the IJK-to-RAS matrix
        if ctData is not None:
          volumeNode = ctData['volumeNode']
          dims = ctData['dimensions']
          ijkToRas = vtk.vtkMatrix4x4()
          volumeNode.GetIJKToRASMatrix(ijkToRas)
          centerIjk = [(dims[0]-1)/2.0, (dims[1]-1)/2.0, (dims[2]-1)/2.0, 1.0]
          centerRas = [0.0, 0.0, 0.0, 1.0]
          ijkToRas.MultiplyPoint(centerIjk, centerRas)
          ctCenterLpsX = -centerRas[0]
          ctCenterLpsY = -centerRas[1]
          ctCenterLpsZ =  centerRas[2]
        else:
          ctCenterLpsX = isoCenterLpsX
          ctCenterLpsY = isoCenterLpsY
          ctCenterLpsZ = isoCenterLpsZ

        logging.info(f"CT center (LPS): [{ctCenterLpsX:.2f}, {ctCenterLpsY:.2f}, {ctCenterLpsZ:.2f}]")
        logging.info(f"Isocenter  (LPS): [{isoCenterLpsX:.2f}, {isoCenterLpsY:.2f}, {isoCenterLpsZ:.2f}]")

        # IMPORTANT: CouchRotation and GantryRotation are placed AT the isocenter in TOPAS
        # World (= DICOM scanner LPS coords), NOT at World origin.
        #
        # With TsDicomPatient, Trans=(0,0,0) puts the CT *center* at World origin — not the
        # CT's natural scanner position. Setting CouchRotation/GantryRotation Trans = iso_LPS
        # anchors all beam rotations at the isocenter in scanner coords. The Patient Trans
        # (relative to CouchRotation) places the CT center at its natural scanner position:
        #   Patient center in World = iso_LPS + (ctCenter_LPS - iso_LPS) = ctCenter_LPS ✓
        #
        # This means TOPAS World == DICOM scanner LPS, so the RT Dose DICOM output will have
        # ImagePositionPatient matching the original CT — and the dose will correctly overlay
        # the CT in Slicer without any position shift.
        #
        # If we had instead placed CouchRotation/GantryRotation at World(0,0,0) (old approach),
        # TOPAS World origin would be at the isocenter, the RT Dose IPP would be shifted by
        # -iso_LPS from the scanner coords, and in Slicer the dose would appear shifted such
        # that it only covered the CT from beam entry to isocenter depth.

        # Couch rotation: centered at isocenter in scanner coords
        f.write("# Couch rotation around isocenter\n")
        f.write('s:Ge/CouchRotation/Type   = "Group"\n')
        f.write('s:Ge/CouchRotation/Parent = "World"\n')
        f.write(f'd:Ge/CouchRotation/TransX = {isoCenterLpsX} mm\n')
        f.write(f'd:Ge/CouchRotation/TransY = {isoCenterLpsY} mm\n')
        f.write(f'd:Ge/CouchRotation/TransZ = {isoCenterLpsZ} mm\n')
        f.write(f'd:Ge/CouchRotation/RotZ   = {couchAngle} deg\n\n')

        # Patient from DICOM CT
        # Trans is in CouchRotation frame: places CT center at ctCenter_LPS in World.
        transX = ctCenterLpsX - isoCenterLpsX
        transY = ctCenterLpsY - isoCenterLpsY
        transZ = ctCenterLpsZ - isoCenterLpsZ
        logging.info(f"Patient Trans (in CouchRotation frame): [{transX:.2f}, {transY:.2f}, {transZ:.2f}]")
        f.write("# Patient from DICOM CT\n")
        f.write('s:Ge/Patient/Type              = "TsDicomPatient"\n')
        f.write('s:Ge/Patient/Parent            = "CouchRotation"\n')
        f.write(f's:Ge/Patient/DicomDirectory    = "{dicomDirectoryTopas}"\n')
        f.write('sv:Ge/Patient/DicomModalityTags = 1 "CT"\n')
        f.write('s:Ge/Patient/ImagingtoMaterialConverter = "Schneider"\n')
        f.write(f'd:Ge/Patient/TransX = {transX} mm\n')
        f.write(f'd:Ge/Patient/TransY = {transY} mm\n')
        f.write(f'd:Ge/Patient/TransZ = {transZ} mm\n')
        f.write('d:Ge/Patient/RotX   = 0. deg\n')
        f.write('d:Ge/Patient/RotY   = 0. deg\n')
        f.write('d:Ge/Patient/RotZ   = 0. deg\n\n')

        # Gantry rotation: centered at isocenter in scanner coords
        f.write("# Gantry rotation around isocenter\n")
        f.write('s:Ge/GantryRotation/Type   = "Group"\n')
        f.write('s:Ge/GantryRotation/Parent = "World"\n')
        f.write(f'd:Ge/GantryRotation/TransX = {isoCenterLpsX} mm\n')
        f.write(f'd:Ge/GantryRotation/TransY = {isoCenterLpsY} mm\n')
        f.write(f'd:Ge/GantryRotation/TransZ = {isoCenterLpsZ} mm\n')
        f.write(f'd:Ge/GantryRotation/RotZ   = {-gantryAngle} deg\n\n')

        # BeamNozzle: RotX=270° redirects beam from TOPAS default +Z to +Y
        # in the GantryRotation frame (toward posterior = from anterior at gantry 0°).
        f.write("# Beam Nozzle (redirects beam from +Z to anterior direction)\n")
        f.write('s:Ge/BeamNozzle/Type   = "Group"\n')
        f.write('s:Ge/BeamNozzle/Parent = "GantryRotation"\n')
        f.write('d:Ge/BeamNozzle/TransX = 0. mm\n')
        f.write('d:Ge/BeamNozzle/TransY = 0. mm\n')
        f.write('d:Ge/BeamNozzle/TransZ = 0. mm\n')
        f.write('d:Ge/BeamNozzle/RotX   = 90.0 deg\n\n')

        # Simple proton beam source
        f.write("# Proton Beam Source\n")
        f.write('s:So/Beam/Type                    = "Beam"\n')
        f.write('s:So/Beam/Component               = "BeamNozzle"\n')
        f.write('s:So/Beam/BeamParticle            = "proton"\n')
        f.write(f'd:So/Beam/BeamEnergy              = {energy} MeV\n')
        f.write('u:So/Beam/BeamEnergySpread        = 0.01\n')
        # Position distribution - use flat distribution matching field size
        halfFieldX = fieldSizeX / 2.0
        halfFieldY = fieldSizeY / 2.0
        f.write('s:So/Beam/BeamPositionDistribution = "Flat"\n')
        f.write('s:So/Beam/BeamPositionCutoffShape = "Rectangle"\n')
        f.write(f'd:So/Beam/BeamPositionCutoffX     = {halfFieldX} mm\n')
        f.write(f'd:So/Beam/BeamPositionCutoffY     = {halfFieldY} mm\n')
        # Angular distribution
        f.write('s:So/Beam/BeamAngularDistribution  = "None"\n')
        # Position beam at SAD distance along beam axis (+Z in nozzle frame)
        f.write(f'd:So/Beam/BeamPositionZ           = {-sad} mm\n')

        f.write(f'i:So/Beam/NumberOfHistoriesInRun  = {numberOfHistories}\n\n')

      # Dose scoring - Output as DICOM RT Dose
      f.write("# Dose Scoring (DICOM RT Dose Output)\n")
      f.write('s:Sc/DoseScorer/Quantity                  = "DoseToMedium"\n')
      f.write('s:Sc/DoseScorer/Component                 = "Patient"\n')
      f.write('s:Sc/DoseScorer/IfOutputFileAlreadyExists = "Overwrite"\n')
      f.write(f's:Sc/DoseScorer/OutputFile                = "{doseFilePathTopas}"\n')
      f.write('s:Sc/DoseScorer/OutputType                = "DICOM"\n')
      # DICOM RT Dose metadata
      f.write(f's:Sc/DoseScorer/DicomPatientDirectory     = "{dicomDirectoryTopas}"\n')
      f.write('s:Sc/DoseScorer/DoseUnits                 = "GY"\n')
      f.write('s:Sc/DoseScorer/DoseType                  = "PHYSICAL"\n')
      f.write('s:Sc/DoseScorer/DoseSummationType         = "PLAN"\n\n')

      # Visualization (optional, disabled by default)
      f.write("# Graphics (disabled for batch mode)\n")
      f.write('b:Gr/Enable = "False"\n\n')

      # Run settings
      f.write("# Run Settings\n")
      f.write('i:Ts/ShowHistoryCountAtInterval = 100000\n')
      f.write('b:Ts/PauseBeforeQuit = "False"\n')

    logging.info(f"Created TOPAS input file: {inputFilePath}")
    return inputFilePath

  #------------------------------------------------------------------------------
  @staticmethod
  def executeTopasSimulation(inputFilePath, topasBinaryPath, workingDirectory, g4dataPath=None, timeout=None):
    """Execute TOPAS simulation

    Args:
      timeout: Maximum time in seconds to wait for the simulation (None = no limit)
    """
    # Check for TOPAS binary existence using command line (cross-platform)
    if TopasDoseEngineUtil.isWindows():
      checkCmd = ['wsl', '--', 'test', '-x', TopasDoseEngineUtil.winToWslPath(topasBinaryPath)]
    else:
      checkCmd = ['test', '-x', topasBinaryPath]
    result = subprocess.run(checkCmd)
    if result.returncode != 0:
      raise RuntimeError(f"TOPAS binary not found or not executable at: {topasBinaryPath}")

    # Change to working directory for simulation
    originalDir = os.getcwd()
    os.chdir(workingDirectory)

    try:
      # Run TOPAS using cross-platform utility
      cmd = [topasBinaryPath, inputFilePath]
      env_vars = None
      if g4dataPath:
        env_vars = {"TOPAS_G4_DATA_DIR": g4dataPath}
      result = TopasDoseEngineUtil.runCommandCrossPlatform(cmd, cwd=workingDirectory, timeout=timeout, env_vars=env_vars)

      if result.returncode != 0:
        logging.error("TOPAS simulation failed.")
        logging.error(f"STDOUT: {result.stdout}")
        logging.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"TOPAS simulation failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

      logging.info("TOPAS simulation completed successfully")
      logging.info(f"Output: {result.stdout}")

    finally:
      os.chdir(originalDir)

    return True

  #------------------------------------------------------------------------------
  @staticmethod
  def findDoseOutputFile(workingDirectory):
    """Find the TOPAS DICOM RT Dose output file.

    Args:
      workingDirectory: Directory containing the dose output file

    Returns:
      Path to the dose DICOM file
    """
    # TOPAS outputs DICOM with the base name we specified
    doseFilePath = os.path.join(workingDirectory, 'dose_output.dcm')

    # Also check for alternative naming patterns TOPAS might use
    if not os.path.exists(doseFilePath):
      doseFiles = glob.glob(os.path.join(workingDirectory, 'dose_output*.dcm'))
      if not doseFiles:
        doseFiles = glob.glob(os.path.join(workingDirectory, '*RTDOSE*.dcm'))
      if doseFiles:
        doseFilePath = doseFiles[0]
      else:
        raise RuntimeError(f"Dose output DICOM file not found in: {workingDirectory}")

    logging.info(f"Found DICOM RT Dose file: {doseFilePath}")
    return doseFilePath

  #------------------------------------------------------------------------------
  @staticmethod
  def findOriginalDicomFiles(volumeNode):
    """Return the list of original DICOM file paths for a CT volume node loaded
    from Slicer's DICOM database, or None if not available.

    Using the original files preserves the Frame of Reference UID that matches
    any associated RT Plan, avoiding the need for IgnoreInconsistentFrameOfReferenceUID.

    Args:
      volumeNode: vtkMRMLScalarVolumeNode loaded from DICOM

    Returns:
      List of file paths if found, None otherwise
    """
    try:
      if not slicer.dicomDatabase.isOpen:
        return None
      shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
      itemID = shNode.GetItemByDataNode(volumeNode)
      frameOfRefUID = shNode.GetItemAttribute(itemID, 'DICOM.FrameOfReferenceUID')
      modality = shNode.GetItemAttribute(itemID, 'DICOM.Modality')
      if not frameOfRefUID or not modality:
        return None
      for patient in slicer.dicomDatabase.patients():
        for study in slicer.dicomDatabase.studiesForPatient(patient):
          for series in slicer.dicomDatabase.seriesForStudy(study):
            files = slicer.dicomDatabase.filesForSeries(series)
            if not files:
              continue
            if (slicer.dicomDatabase.fileValue(files[0], '0020,0052') == frameOfRefUID and
                slicer.dicomDatabase.fileValue(files[0], '0008,0060') == modality):
              logging.info(f"Found {len(files)} original DICOM CT files in Slicer database")
              return list(files)
      return None
    except Exception as e:
      logging.warning(f"Could not retrieve original DICOM files: {e}")
      return None

  #------------------------------------------------------------------------------
  @staticmethod
  def runTopasSimulation(ctData, beamProperties, topasBinaryPath, g4dataPath=None, topasDirectoryPath=None, planFilePath=None, timeout=None):
    """Complete TOPAS simulation workflow using DICOM input.

    Args:
      ctData: CT data dictionary from extractCTData()
      beamProperties: Beam properties dictionary from extractBeamProperties()
      topasBinaryPath: Path to TOPAS executable
      g4dataPath: Path to Geant4 data directory (optional)
      topasDirectoryPath: Path to TOPAS installation directory
      planFilePath: Path to RT Ion Plan DICOM file (optional, enables TsRTIonSource)
      timeout: Maximum time in seconds to wait for the simulation (None = no limit)

    Returns:
      Tuple of (doseFilePath, workingDirectory) where doseFilePath is the path
      to the DICOM RT Dose file and workingDirectory is the temp directory to
      clean up after loading.
    """
    workingDirectory = tempfile.mkdtemp(prefix='topas_dose_')

    # Try to reuse the original DICOM files from Slicer's database.
    # This preserves the Frame of Reference UID so it matches any RT Plan.
    # Copy only the CT files into a clean subdirectory even when originals are found —
    # the source directory may contain mixed DICOM modalities (CT + RT Struct/Plan/Dose)
    # which causes TOPAS TsDicomPatient to fail with "Failed to sort files".
    dicomDirectory = os.path.join(workingDirectory, 'dicom_ct')
    os.makedirs(dicomDirectory, exist_ok=True)
    originalFiles = TopasDoseEngineUtil.findOriginalDicomFiles(ctData['volumeNode'])
    if originalFiles:
      logging.info(f"Copying {len(originalFiles)} original CT DICOM files to clean temp directory...")
      for f in originalFiles:
        shutil.copy2(f, dicomDirectory)
      logging.info("Done copying — skipping CT export step.")
    else:
      logging.info("No original DICOM found — exporting CT as DICOM series for TOPAS...")
      TopasDoseEngineUtil.exportCTAsDicomSeries(ctData, dicomDirectory)

    # Create TOPAS input file referencing DICOM
    logging.info("Creating TOPAS input file (DICOM)...")
    inputFilePath = TopasDoseEngineUtil.createTopasInputFileDicom(
      dicomDirectory=dicomDirectory,
      beamProperties=beamProperties,
      workingDirectory=workingDirectory,
      topasDirectory=topasDirectoryPath,
      ctData=ctData,
      planFilePath=planFilePath
    )

    # Execute simulation
    logging.info("Executing TOPAS simulation...")
    TopasDoseEngineUtil.executeTopasSimulation(inputFilePath, topasBinaryPath, workingDirectory, g4dataPath, timeout)

    # Find the dose output file
    logging.info("Locating DICOM RT Dose output...")
    doseFilePath = TopasDoseEngineUtil.findDoseOutputFile(workingDirectory)

    return doseFilePath, workingDirectory

  #------------------------------------------------------------------------------
  @staticmethod
  def loadDoseResultAsVolume(resultDoseVolumeNode, doseFilePath, workingDirectory):
    """Load TOPAS dose result into Slicer volume node using SlicerRT's DICOM RT import.

    Uses the same DicomRtImportExport logic that runs when importing through the
    DICOM module, so DoseGridScaling and all RT-specific metadata are handled properly.

    Args:
      resultDoseVolumeNode: Slicer volume node to store the dose
      doseFilePath: Path to the DICOM RT Dose file
      workingDirectory: Temporary working directory to clean up after loading
    """
    try:
      # Use SlicerRT's DicomRtImportExport logic (same path as manual DICOM import)
      vtkFileList = vtk.vtkStringArray()
      vtkFileList.InsertNextValue(doseFilePath)

      loadablesCollection = vtk.vtkCollection()
      slicer.modules.dicomrtimportexport.logic().ExamineForLoad(vtkFileList, loadablesCollection)

      if loadablesCollection.GetNumberOfItems() == 0:
        raise RuntimeError(f"DicomRtImportExport could not recognize dose file: {doseFilePath}")

      # Track existing volume nodes to identify the newly loaded one
      existingNodeIDs = set()
      for i in range(slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLScalarVolumeNode')):
        existingNodeIDs.add(slicer.mrmlScene.GetNthNodeByClass(i, 'vtkMRMLScalarVolumeNode').GetID())

      # Load through the RT import logic
      vtkLoadable = loadablesCollection.GetItemAsObject(0)
      success = slicer.modules.dicomrtimportexport.logic().LoadDicomRT(vtkLoadable)
      if not success:
        raise RuntimeError(f"Failed to load dose DICOM file via DicomRtImportExport: {doseFilePath}")

      # Find the newly created dose volume node
      tempDoseNode = None
      for i in range(slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLScalarVolumeNode')):
        node = slicer.mrmlScene.GetNthNodeByClass(i, 'vtkMRMLScalarVolumeNode')
        if node.GetID() not in existingNodeIDs:
          tempDoseNode = node
          break

      if not tempDoseNode:
        raise RuntimeError("Could not find loaded dose volume node after DicomRtImportExport")

      # Copy geometry and voxel data
      resultDoseVolumeNode.CopyOrientation(tempDoseNode)
      doseImageDataCopy = vtk.vtkImageData()
      doseImageDataCopy.DeepCopy(tempDoseNode.GetImageData())
      resultDoseVolumeNode.SetAndObserveImageData(doseImageDataCopy)

      # Transfer the display node from tempDoseNode to resultDoseVolumeNode.
      # DicomRtImportExport fully configures it (window/level, colormap, dose settings).
      # Display nodes are independent scene objects — removing tempDoseNode does NOT
      # remove its display node, so the transfer is safe.
      tempDisplayNode = tempDoseNode.GetDisplayNode()
      if tempDisplayNode:
        resultDoseVolumeNode.AddAndObserveDisplayNodeID(tempDisplayNode.GetID())
      else:
        resultDoseVolumeNode.CreateDefaultDisplayNodes()

      # Diagnostic logging
      doseRange = doseImageDataCopy.GetScalarRange()
      logging.info(f"Dose scalar type: {doseImageDataCopy.GetScalarTypeAsString()}, dims: {doseImageDataCopy.GetDimensions()}")
      logging.info(f"Dose range: min={doseRange[0]:.6g}, max={doseRange[1]:.6g}, total voxels={doseImageDataCopy.GetNumberOfPoints()}")
      logging.info(f"Result node origin: {resultDoseVolumeNode.GetOrigin()}")
      logging.info(f"Result node spacing: {resultDoseVolumeNode.GetSpacing()}")
      logging.info(f"Dose volume loaded successfully from: {doseFilePath}")

      # Remove the temporary node — its display node stays in the scene,
      # now referenced by resultDoseVolumeNode
      slicer.mrmlScene.RemoveNode(tempDoseNode)

    finally:
      #TODO: Re-enable cleanup after testing
      logging.info(f"Temporary directory kept for debugging: {workingDirectory}")
      #import shutil
      #if os.path.exists(workingDirectory):
      #  shutil.rmtree(workingDirectory)
      #  logging.info(f"Cleaned up temporary directory: {workingDirectory}")

  #------------------------------------------------------------------------------
  @staticmethod
  def isWindows():
    """Detect if running on Windows."""
    return os.name == 'nt'

  #------------------------------------------------------------------------------
  @staticmethod
  def winToWslPath(winPath):
    """Convert a Windows path to a WSL path."""
    import re
    # Example: C:\Users\User\file.txt -> /mnt/c/Users/User/file.txt
    drive, path = os.path.splitdrive(winPath)
    if not drive:
      return winPath.replace('\\', '/')
    driveLetter = drive[0].lower()
    wslPath = f"/mnt/{driveLetter}{path.replace('\\', '/')}"
    return wslPath

  #------------------------------------------------------------------------------
  @staticmethod
  def runCommandCrossPlatform(cmd, cwd=None, captureOutput=True, text=True, timeout=None, env_vars=None):
    """Run a command in WSL if on Windows, or natively otherwise. Optionally export env_vars in WSL."""
    # Ensure cwd is a valid directory if provided
    if cwd and not os.path.isdir(cwd):
      raise NotADirectoryError(f"The working directory '{cwd}' does not exist or is not a directory.")
    if TopasDoseEngineUtil.isWindows():
      # Convert all paths in cmd to WSL format if they are absolute Windows paths
      def convertArg(arg):
        if isinstance(arg, str) and (os.path.isabs(arg) and ':' in arg):
          return TopasDoseEngineUtil.winToWslPath(arg)
        return arg
      cmdWsl = [convertArg(a) for a in cmd]
      # Prepare export string if env_vars provided
      export_str = ''
      if env_vars:
        export_str = ' '.join([f"export {k}='{v}';" for k, v in env_vars.items()])
      if cwd:
        wslCwd = TopasDoseEngineUtil.winToWslPath(cwd)
        cmdWsl = ['wsl', '--', 'bash', '-c', f"cd '{wslCwd}' && {export_str} {' '.join(cmdWsl)}"]
        return subprocess.run(cmdWsl, capture_output=captureOutput, text=text, timeout=timeout)
      else:
        cmdWsl = ['wsl', '--', 'bash', '-c', f"{export_str} {' '.join(cmdWsl)}"]
        return subprocess.run(cmdWsl, capture_output=captureOutput, text=text, timeout=timeout)
    else:
      env = {**os.environ, **env_vars} if env_vars else None
      return subprocess.run(cmd, cwd=cwd, capture_output=captureOutput, text=text, timeout=timeout, env=env)
