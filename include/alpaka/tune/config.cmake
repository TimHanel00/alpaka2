# strategyConfig.cmake

# === STRATEGY SELECTION ===
set(STRATEGY "randomSearch" CACHE STRING "Tuning strategy to use")
set_property(CACHE STRATEGY PROPERTY STRINGS randomSearch exhaustiveSearch simulatedAnnealing randomSample)

# Clear all strategy flags (defensive)
unset(strategy_randomSearch CACHE)
unset(strategy_exhaustiveSearch CACHE)
unset(strategy_simulatedAnnealing CACHE)
unset(strategy_randomSample CACHE)

# Define the selected strategy
if (STRATEGY STREQUAL "randomSearch")
    add_compile_definitions(strategy_randomSearch)
elseif (STRATEGY STREQUAL "exhaustiveSearch")
    add_compile_definitions(strategy_exhaustiveSearch)
elseif (STRATEGY STREQUAL "simulatedAnnealing")
    add_compile_definitions(strategy_simulatedAnnealing)
elseif (STRATEGY STREQUAL "randomSample")
    add_compile_definitions(strategy_randomSample)
else ()
    message(FATAL_ERROR "Unknown STRATEGY: ${STRATEGY}")
endif ()

# === OPTIONAL FEATURE FLAGS ===

# This flag defaults to OFF if not specified
option(ExhaustiveSearchRandomInitialization "Enable random initialization for exhaustive search" OFF)

if (ExhaustiveSearchRandomInitialization)
    add_definitions(-DExhaustiveSearchRandomInitialization)
    add_compile_definitions(ExhaustiveSearchRandomInitialization)
endif ()
