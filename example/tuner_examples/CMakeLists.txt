#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.25)

project("alpakaExamples" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################
if (alpaka_TUNER)
    add_subdirectory("babelstream/")
    add_subdirectory("heatEqu2DWithFrame/")
endif ()