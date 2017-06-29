#pragma once
#ifndef FORPY_UTIL_SERIALIZATION_EIGEN_H_
#define FORPY_UTIL_SERIALIZATION_EIGEN_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>

#include <Eigen/Dense>
#include "../../global.h"

// Eigen serialization helper.
// (c.f. https://stackoverflow.com/questions/22884216/serializing-eigenmatrix-using-cereal-library)
namespace cereal {

  // Save Eigen matrix in binary archive.
  template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
  typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
  save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m) {
    Eigen::Index rows = m.rows();
    Eigen::Index cols = m.cols();
    ar(CEREAL_NVP(rows));
    ar(CEREAL_NVP(cols));
    ar(binary_data(m.data(), static_cast<size_t>(rows * cols *
                                                 sizeof(_Scalar))));
  }
  // Save Eigen matrix in non-binary archive.
  template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
  typename std::enable_if<!traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
  save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m) {
    Eigen::Index rows = m.rows();
    Eigen::Index cols = m.cols();
    ar(CEREAL_NVP(rows));
    ar(CEREAL_NVP(cols));
    for (Eigen::Index i = 0; i < rows; ++i) {
      for (Eigen::Index j = 0; j < cols; ++j) {
        ar(m(i, j));
      }
    }
  }
  // Load Eigen matrix from binary archive.
  template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
  typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
  load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m) {
    Eigen::Index rows;
    Eigen::Index cols;
    ar(CEREAL_NVP(rows));
    ar(CEREAL_NVP(cols));
    m.resize(rows, cols);
    ar(binary_data(m.data(), static_cast<size_t>(rows * cols *
                                                 sizeof(_Scalar))));
  }
  // Load Eigen matrix from non-binary archive.
  template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
  typename std::enable_if<!traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m) {
    Eigen::Index rows;
    Eigen::Index cols;
    ar(CEREAL_NVP(rows));
    ar(CEREAL_NVP(cols));
    m.resize(rows, cols);
    for (Eigen::Index i = 0; i < rows; ++i) {
      for (Eigen::Index j = 0; j < cols; ++j) {
        ar(m(i, j));
      }
    }
  }
} // namespace cereal

#endif // FORPY_UTIL_SERIALIZATION_EIGEN_H_
