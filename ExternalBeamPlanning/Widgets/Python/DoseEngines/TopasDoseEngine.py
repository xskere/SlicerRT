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

    # Load paths from application settings
    self.loadPathsFromApplicationSettings()

  #------------------------------------------------------------------------------
  def defineBeamParameters(self):
    """Define beam parameters that can be configured"""
    tabName = "Topas parameters"

    # Energy parameter
    self.scriptedEngine.addBeamParameterSpinBox(
      tabName, 'energy', 'Beam energy',
      'Energy of the proton beam in MeV',
      1.0, 250.0, 100.0, 1.0, 1)

    # Number of histories parameter
    self.scriptedEngine.addBeamParameterSpinBox(
      tabName, 'numberOfHistories', 'Number of particles',
      'Number of particles to simulate (more particles = better statistics but slower)',
      100000, 100000000, 1000000, 500000, 0)

  #------------------------------------------------------------------------------
  def loadPathsFromApplicationSettings(self):
    """Load TOPAS paths from application settings"""
    settings = slicer.app.userSettings()
    self.topasDirectoryPath = settings.value('TopasDoseEngine/TopasDirectory', self.topasDirectoryPath)
    self.topasBinaryPath = settings.value('TopasDoseEngine/TopasBinary', self.topasBinaryPath)
    self.rtIonPlanFilePath = settings.value('TopasDoseEngine/RTIonPlanFile', self.rtIonPlanFilePath)

  #------------------------------------------------------------------------------
  def savePathsInApplicationSettings(self, beamNode=None):
    """Save TOPAS paths to application settings"""
    settings = slicer.app.userSettings()
    settings.setValue('TopasDoseEngine/TopasDirectory', self.topasDirectoryPath)
    settings.setValue('TopasDoseEngine/TopasBinary', self.topasBinaryPath)
    settings.setValue('TopasDoseEngine/RTIonPlanFile', self.rtIonPlanFilePath)

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

      logging.info(f"Custom beam parameters: energy={beamProperties['energy']} MeV,"
                   f"histories={beamProperties['numberOfHistories']}")

      # Determine RT Ion Plan file path (if available)
      planFilePath = None
      if self.rtIonPlanFilePath and os.path.exists(self.rtIonPlanFilePath):
        planFilePath = self.rtIonPlanFilePath
        logging.info(f"Using RT Ion Plan file: {planFilePath}")

      # Create and run TOPAS simulation using static utility
      logging.info("Running TOPAS simulation...")
      doseFilePath, workingDirectory = TopasDoseEngineUtil.runTopasSimulation(
        ctData=ctData,
        beamProperties=beamProperties,
        topasBinaryPath=self.topasBinaryPath,
        g4dataPath=self.g4dataPath,
        topasDirectoryPath=self.topasDirectoryPath,
        planFilePath=planFilePath
      )

      # Load result as volume in Slicer using standard DICOM loading
      logging.info("Loading dose result...")
      TopasDoseEngineUtil.loadDoseResultAsVolume(resultDoseVolumeNode, doseFilePath, workingDirectory)

      logging.info("TOPAS dose calculation completed successfully!")
      return True

    except Exception as e:
      logging.error(f"Error in TOPAS dose calculation: {str(e)}")
      raise e
