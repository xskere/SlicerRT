import os
import slicer
from DoseEngines import AbstractScriptedDoseEngine
from Python.TopasDoseEngineUtil import TopasDoseEngineUtil
import logging

#------------------------------------------------------------------------------
#
# TopasDoseEngine
#
#------------------------------------------------------------------------------
class TopasDoseEngine(AbstractScriptedDoseEngine):
  """ Topas python dose engine
  """

  #------------------------------------------------------------------------------
  def __init__(self, scriptedEngine):
    scriptedEngine.name = 'Topas'
    AbstractScriptedDoseEngine.__init__(self, scriptedEngine)

    # Define initial defaults for parameters that are stored in application settings
    self.topasDirectoryPath = '/home/xskere/topas'
    self.topasBinaryPath = '/home/xskere/topas/bin/topas'
    self.g4dataPath = '/home/xskere/G4Data'
    self.rtIonPlanFilePath = ''  # Optional: Path to RT Ion Plan DICOM file for TsRTIonSource
    self.machineDataFilePath = ''  # Optional: Path to .mat file with energy-dependent spot sigma and SAD

    # Load paths from application settings
    self.loadPathsFromApplicationSettings()

  #------------------------------------------------------------------------------
  def defineBeamParameters(self):
    """Define beam parameters that can be configured"""
    tabName = "Topas parameters"

    # Radiation mode (beam particle)
    self.scriptedEngine.addBeamParameterComboBox(
      tabName, 'radiationMode', 'Radiation mode',
      'Particle type used for the simulation',
      ['proton', 'neutron', 'gamma', 'e-', 'e+'], 0)

    # Energy parameter
    self.scriptedEngine.addBeamParameterSpinBox(
      tabName, 'energy', 'Beam energy',
      'Energy of the beam in MeV',
      1.0, 250.0, 100.0, 1.0, 1)

    # Number of histories parameter
    self.scriptedEngine.addBeamParameterSpinBox(
      tabName, 'numberOfHistories', 'Number of particles',
      'Number of particles to simulate (more particles = better statistics but slower)',
      100000, 100000000, 1000000, 500000, 0)

    # ParticlesPerHistory: downsampling factor — histories/spot = spot_MU / ParticlesPerHistory
    # Values < 1 increase histories (better statistics); values > 1 decrease them (faster)
    self.scriptedEngine.addBeamParameterSpinBox(
      tabName, 'particlesPerHistory', 'Particles per history',
      'Downsampling factor for TsRTIonSource. Values < 1 increase histories per spot for better statistics.',
      0.0001, 1000000.0, 1.0, 0.1, 4)

    # Path parameters
    self.scriptedEngine.addBeamParameterLineEdit(
      tabName, 'topasDirectory', 'TOPAS directory:',
      'Path to the TOPAS installation directory', self.topasDirectoryPath)

    self.scriptedEngine.addBeamParameterLineEdit(
      tabName, 'topasBinary', 'TOPAS binary:',
      'Path to the TOPAS executable', self.topasBinaryPath)

    self.scriptedEngine.addBeamParameterLineEdit(
      tabName, 'g4DataDirectory', 'Geant4 data directory:',
      'Path to the Geant4 data directory (G4DATAFILES)', self.g4dataPath)

    self.scriptedEngine.addBeamParameterLineEdit(
      tabName, 'rtIonPlanFile', 'RT Ion Plan file (optional):',
      'Path to a DICOM RT Ion Plan file for TsRTIonSource. Leave empty to use the simple beam source.',
      self.rtIonPlanFilePath)

    self.scriptedEngine.addBeamParameterLineEdit(
      tabName, 'machineDataFile', 'Machine data file (optional):',
      'Path to a .mat file with energy-dependent spot sigma and SAD (e.g. TROTS MachineData.mat).',
      self.machineDataFilePath)

  #------------------------------------------------------------------------------
  def loadPathsFromApplicationSettings(self):
    """Load TOPAS paths from application settings"""
    settings = slicer.app.userSettings()
    self.topasDirectoryPath = settings.value('TopasDoseEngine/TopasDirectory', self.topasDirectoryPath)
    self.topasBinaryPath = settings.value('TopasDoseEngine/TopasBinary', self.topasBinaryPath)
    self.g4dataPath = settings.value('TopasDoseEngine/G4DataDirectory', self.g4dataPath)
    self.rtIonPlanFilePath = settings.value('TopasDoseEngine/RTIonPlanFile', self.rtIonPlanFilePath)
    self.machineDataFilePath = settings.value('TopasDoseEngine/MachineDataFile', self.machineDataFilePath)

  #------------------------------------------------------------------------------
  def savePathsInApplicationSettings(self, beamNode=None):
    """Save TOPAS paths to application settings"""
    settings = slicer.app.userSettings()
    settings.setValue('TopasDoseEngine/TopasDirectory', self.topasDirectoryPath)
    settings.setValue('TopasDoseEngine/TopasBinary', self.topasBinaryPath)
    settings.setValue('TopasDoseEngine/G4DataDirectory', self.g4dataPath)
    settings.setValue('TopasDoseEngine/RTIonPlanFile', self.rtIonPlanFilePath)
    settings.setValue('TopasDoseEngine/MachineDataFile', self.machineDataFilePath)

  #------------------------------------------------------------------------------
  def calculateDoseUsingEngine(self, beamNode, resultDoseVolumeNode):
    """Main method to calculate dose using TOPAS.

    Uses TsRTIonSource from TOPAS dicom-interface if rtIonPlanFilePath is set,
    otherwise falls back to a simple proton beam source.
    """
    try:
      logging.info("Starting TOPAS dose calculation...")

      # Get plan node from beam
      planNode = beamNode.GetParentPlanNode()
      if not planNode:
        raise RuntimeError("No plan node found for beam")

      # Extract CT data using static utility
      logging.info("Extracting CT data...")
      ctData = TopasDoseEngineUtil.extractCTData(planNode)

      # Extract beam properties using static utility
      logging.info("Extracting beam properties...")
      beamProperties = TopasDoseEngineUtil.extractBeamProperties(beamNode)

      # Override with custom beam parameters from UI
      beamProperties['energy'] = self.scriptedEngine.doubleParameter(beamNode, 'energy')
      beamProperties['numberOfHistories'] = int(self.scriptedEngine.doubleParameter(beamNode, 'numberOfHistories'))
      beamProperties['particlesPerHistory'] = self.scriptedEngine.doubleParameter(beamNode, 'particlesPerHistory')

      _radiationModes = ['proton', 'neutron', 'gamma', 'e-', 'e+']
      radiationModeIdx = int(self.scriptedEngine.parameter(beamNode, 'radiationMode') or 0)
      beamProperties['radiationMode'] = _radiationModes[radiationModeIdx]

      topasDirectoryPath = self.scriptedEngine.parameter(beamNode, 'topasDirectory')
      topasBinaryPath = self.scriptedEngine.parameter(beamNode, 'topasBinary')
      g4dataPath = self.scriptedEngine.parameter(beamNode, 'g4DataDirectory')
      rtIonPlanFilePath = self.scriptedEngine.parameter(beamNode, 'rtIonPlanFile')
      machineDataFilePath = self.scriptedEngine.parameter(beamNode, 'machineDataFile')
      beamProperties['machineDataFile'] = machineDataFilePath if machineDataFilePath and os.path.exists(machineDataFilePath) else None

      # Persist paths to application settings so they survive session restarts
      self.topasDirectoryPath = topasDirectoryPath
      self.topasBinaryPath = topasBinaryPath
      self.g4dataPath = g4dataPath
      self.rtIonPlanFilePath = rtIonPlanFilePath
      self.machineDataFilePath = machineDataFilePath
      self.savePathsInApplicationSettings()

      logging.info(f"Custom beam parameters: energy={beamProperties['energy']} MeV,"
                   f"histories={beamProperties['numberOfHistories']}")

      # Determine RT Ion Plan file path (if available)
      planFilePath = None
      if rtIonPlanFilePath and os.path.exists(rtIonPlanFilePath):
        planFilePath = rtIonPlanFilePath
        logging.info(f"Using RT Ion Plan file: {planFilePath}")

      # Create and run TOPAS simulation using static utility
      logging.info("Running TOPAS simulation...")
      doseFilePath, workingDirectory = TopasDoseEngineUtil.runTopasSimulation(
        ctData=ctData,
        beamProperties=beamProperties,
        topasBinaryPath=topasBinaryPath,
        g4dataPath=g4dataPath,
        topasDirectoryPath=topasDirectoryPath,
        planFilePath=planFilePath
      )

      # Load result as volume in Slicer using standard DICOM loading
      logging.info("Loading dose result...")
      TopasDoseEngineUtil.loadDoseResultAsVolume(resultDoseVolumeNode, doseFilePath, workingDirectory)

      logging.info("TOPAS dose calculation completed successfully!")
      return str()

    except Exception as e:
      logging.error(f"Error in TOPAS dose calculation: {str(e)}")
      raise e
