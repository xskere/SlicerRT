import os
import tempfile
import subprocess
import vtk
import slicer
import logging    
import slicer
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

    sourcePosition = [0.0, 0.0, 0.0]
    beamNode.GetSourcePosition(sourcePosition)
    beamProperties['sourcePosition'] = sourcePosition
    
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
    logging.info(f"Calculated field size X: {beamProperties['fieldSizeX']} cm")
    beamProperties['fieldSizeY'] = abs(y2 - y1)
    logging.info(f"Calculated field size Y: {beamProperties['fieldSizeY']} cm")

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
      f.write('sv:Ph/Default/Modules = 1 "g4em-standard_opt4"\n\n')

      # World geometry
      f.write("# World\n")
      f.write('s:Ge/World/Type     = "TsBox"\n')
      f.write('s:Ge/World/Material = "G4_AIR"\n')
      f.write('d:Ge/World/HLX      = 2.0 m\n')
      f.write('d:Ge/World/HLY      = 2.0 m\n')
      f.write('d:Ge/World/HLZ      = 2.0 m\n\n')

      # Patient geometry from DICOM CT
      f.write("# Patient from DICOM CT\n")
      f.write('s:Ge/Patient/Type              = "TsDicomPatient"\n')
      f.write('s:Ge/Patient/Parent            = "World"\n')
      f.write(f's:Ge/Patient/DicomDirectory    = "{dicomDirectoryTopas}"\n')
      f.write('sv:Ge/Patient/DicomModalityTags = 1 "CT"\n')
      f.write('s:Ge/Patient/ImagingtoMaterialConverter = "Schneider"\n')

      # Patient positioning: translate so the isocenter lands at World origin.
      # TsDicomPatient centers the CT bounding box at the component position (Trans).
      # With Trans=0 the CT center is at World origin, so:
      #   Trans = CT_center_LPS - Isocenter_LPS
      # Isocenter from Slicer RAS → DICOM/LPS: flip X and Y
      isoCenterLpsX = -isocenter[0]
      isoCenterLpsY = -isocenter[1]
      isoCenterLpsZ =  isocenter[2]
      if ctData is not None:
        # Compute CT volume center via the IJK-to-RAS matrix
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
        transX = ctCenterLpsX - isoCenterLpsX
        transY = ctCenterLpsY - isoCenterLpsY
        transZ = ctCenterLpsZ - isoCenterLpsZ
        logging.info(f"CT center (LPS): [{ctCenterLpsX:.2f}, {ctCenterLpsY:.2f}, {ctCenterLpsZ:.2f}]")
        logging.info(f"Isocenter  (LPS): [{isoCenterLpsX:.2f}, {isoCenterLpsY:.2f}, {isoCenterLpsZ:.2f}]")
        logging.info(f"Patient Trans: [{transX:.2f}, {transY:.2f}, {transZ:.2f}]")
      else:
        # Fallback: assume CT is already centered at World origin
        transX = -isoCenterLpsX
        transY = -isoCenterLpsY
        transZ = -isoCenterLpsZ
      f.write(f'd:Ge/Patient/TransX = {transX} mm\n')
      f.write(f'd:Ge/Patient/TransY = {transY} mm\n')
      f.write(f'd:Ge/Patient/TransZ = {transZ} mm\n')
      f.write('d:Ge/Patient/RotX   = 0. deg\n')
      f.write('d:Ge/Patient/RotY   = 0. deg\n')
      f.write('d:Ge/Patient/RotZ   = 0. deg\n\n')

      f.write("# Gantry rotation around patient long axis (Z/superior)\n")
      f.write('s:Ge/GantryRotation/Type   = "Group"\n')
      f.write('s:Ge/GantryRotation/Parent = "World"\n')
      f.write('d:Ge/GantryRotation/TransX = 0. mm\n')
      f.write('d:Ge/GantryRotation/TransY = 0. mm\n')
      f.write('d:Ge/GantryRotation/TransZ = 0. mm\n')
      f.write(f'd:Ge/GantryRotation/RotZ   = {gantryAngle} deg\n\n')

      f.write("# Beam Nozzle (redirects beam from +Z to anterior direction)\n")
      f.write('s:Ge/BeamNozzle/Type   = "Group"\n')
      f.write('s:Ge/BeamNozzle/Parent = "GantryRotation"\n')
      f.write('d:Ge/BeamNozzle/TransX = 0. mm\n')
      f.write('d:Ge/BeamNozzle/TransY = 0. mm\n')
      f.write('d:Ge/BeamNozzle/TransZ = 0. mm\n')

      # RT Ion Source configuration
      f.write("# RT Ion Beam Source\n")
      if planFilePathTopas:
        # Use DICOM RT Ion Plan file
        f.write('s:So/Beam/Type                    = "TsRTIonSource"\n')
        f.write('s:So/Beam/Component               = "BeamNozzle"\n')
        f.write(f's:So/Beam/File                    = "{planFilePathTopas}"\n')
        f.write(f's:So/Beam/imgdirectory            = "{dicomDirectoryTopas}"\n')
        f.write('i:So/Beam/beamnumber              = 1\n')
        f.write(f'd:So/Beam/sid                     = {sad} mm\n')
        f.write(f'd:So/Beam/RotGantry               = {gantryAngle} deg\n')
        f.write(f'd:So/Beam/RotCollimator           = {collimatorAngle} deg\n')
        f.write(f'd:So/Beam/RotPatientSupport       = {couchAngle} deg\n')
        f.write('d:So/Beam/ShiftX                  = 0. mm\n')
        f.write('d:So/Beam/ShiftY                  = 0. mm\n')
        f.write('d:So/Beam/ShiftZ                  = 0. mm\n')
        f.write('u:So/Beam/particlesperhistory     = 1\n')
      else:
        # Fallback: Create a simple proton beam source without RT Plan file
        f.write('s:So/Beam/Type                    = "Beam"\n')
        f.write('s:So/Beam/Component               = "BeamNozzle"\n')
        f.write('s:So/Beam/BeamParticle            = "proton"\n')
        f.write(f'd:So/Beam/BeamEnergy              = {energy} MeV\n')
        f.write('u:So/Beam/BeamEnergySpread        = 0.5\n')
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

    # Export CT as DICOM
    logging.info("Exporting CT as DICOM series for TOPAS...")
    dicomDirectory = os.path.join(workingDirectory, 'dicom_ct')
    os.makedirs(dicomDirectory, exist_ok=True)
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

      # Copy image data and geometry to the result node
      resultDoseVolumeNode.CopyOrientation(tempDoseNode)
      doseImageDataCopy = vtk.vtkImageData()
      doseImageDataCopy.DeepCopy(tempDoseNode.GetImageData())
      resultDoseVolumeNode.SetAndObserveImageData(doseImageDataCopy)
      resultDoseVolumeNode.SetName("TOPAS_Dose")

      logging.info(f"Dose volume loaded successfully from: {doseFilePath}")

      # Remove temporary volume node
      slicer.mrmlScene.RemoveNode(tempDoseNode)

    finally:
      import shutil
      if os.path.exists(workingDirectory):
        shutil.rmtree(workingDirectory)
        logging.info(f"Cleaned up temporary directory: {workingDirectory}")

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
      return subprocess.run(cmd, cwd=cwd, capture_output=captureOutput, text=text, timeout=timeout)
