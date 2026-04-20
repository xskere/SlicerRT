#ifndef __qSlicerIpoptOptimizer_h
#define __qSlicerIpoptOptimizer_h

// ExternalBeamPlanning includes
#include "qSlicerAbstractPlanOptimizer.h"
#include "qSlicerExternalBeamPlanningModuleWidgetsExport.h"

// Qt includes
#include <QMap>
#include <QVariant>
#include <QString>

// IPOPT includes
#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"
#include "IpSolveStatistics.hpp"

// STL includes
#include <vector>
#include <map>
#include <functional>
#include <memory>

#include <vtkSmartPointer.h>

using namespace Ipopt;

class qSlicerIpoptOptimizerPrivate;
class vtkMRMLRTPlanNode;
class vtkMRMLRTObjectiveNode;
class vtkMRMLScalarVolumeNode;

/// \brief Interior Point Optimizer using IPOPT
///
/// Integrates the IPOPT solver with the SlicerRT plan optimization framework.
/// Inherits from qSlicerAbstractPlanOptimizer so it can be registered with the
/// plugin handler and invoked through the ExternalBeamPlanning module UI.
class Q_SLICER_MODULE_EXTERNALBEAMPLANNING_WIDGETS_EXPORT qSlicerIpoptOptimizer
  : public qSlicerAbstractPlanOptimizer
{
  Q_OBJECT

public:
  typedef qSlicerAbstractPlanOptimizer Superclass;

  explicit qSlicerIpoptOptimizer(QObject* parent = nullptr);
  ~qSlicerIpoptOptimizer() override;

  static const QString NAME;
  static const QString SHORT_NAME;
  static const bool GPU_COMPATIBLE;
  static const bool ALLOW_KEYBOARD_CANCEL;

  static const double DEFAULT_ABS_OBJ_TOL;
  static const int DEFAULT_MAX_ITER;
  static const double DEFAULT_MAX_TIME;

  /// Register all available objective and constraint types.
  void setAvailableObjectives() override;

public:
  using Array = std::vector<double>;
  using ObjectiveFunction = std::function<double(const Array&)>;
  using GradientFunction = std::function<Array(const Array&)>;

  struct Options {
    int print_level = 5;
    QString print_user_options = "no";
    QString print_options_documentation = "no";
    double tol = 1e-10;
    double dual_inf_tol = 1e-4;
    double constr_viol_tol = 1e-4;
    double compl_inf_tol = 1e-4;
    int acceptable_iter = 5;
    double acceptable_tol = DEFAULT_ABS_OBJ_TOL;
    double acceptable_constr_viol_tol = 1e-2;
    double acceptable_dual_inf_tol = 1e10;
    double acceptable_compl_inf_tol = 1e10;
    double acceptable_obj_change_tol = 1e-4;
    int max_iter = DEFAULT_MAX_ITER;
    double max_cpu_time = DEFAULT_MAX_TIME;
    QString mu_strategy = "adaptive";
    QString hessian_approximation = "limited-memory";
    int limited_memory_max_history = 20;
    QString limited_memory_initialization = "scalar2";
    QString linear_solver = "mumps";
    QString print_timing_statistics = "yes";
  };

  struct Result {
    Array solution;
    ApplicationReturnStatus status;
    double final_objective_value = 0.0;
    int iteration_count = 0;
    bool success = false;
  };

public slots:
  void setMaxIterations(int maxIter);
  void setMaxTime(double maxTime);
  void setAbsoluteObjectiveTolerance(double tol);

  void setObjectiveFunction(ObjectiveFunction func);
  void setGradientFunction(GradientFunction func);

public:
  Result solveProblem(const Array& x0);

  Options getOptions() const;
  void setOptions(const Options& options);
  void setOption(const QString& key, const QVariant& value);

  Result getLastResult() const;

signals:
  void iterationUpdate(int iteration, double objectiveValue);
  void optimizationCompleted(bool success, const QString& message);

protected:
  /// Called by the ExternalBeamPlanning module to run the optimizer on a plan.
  QString optimizePlanUsingOptimizer(
    vtkMRMLRTPlanNode* planNode,
    std::vector<vtkSmartPointer<vtkMRMLRTObjectiveNode>> objectives,
    vtkMRMLScalarVolumeNode* resultOptimizationVolumeNode) override;

private:
  QScopedPointer<qSlicerIpoptOptimizerPrivate> d_ptr;
  Q_DECLARE_PRIVATE(qSlicerIpoptOptimizer);
  Q_DISABLE_COPY(qSlicerIpoptOptimizer);

  class IpoptProblem;
  friend class IpoptProblem;
};

#endif
