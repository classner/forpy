// Macros to deal with the type overloads in a convenient way.
#define FORPY_UNPACK(...) __VA_ARGS__

/********************** Type instantiation combinations ***********************/
// Only annotation type.
#define FORPY_GEN_FUNCHEAD_AT(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF)     \
  DOC PREF RETVAL(float) FNAME(FPARAMS(float)) MOD SUFF                 \
  DOC PREF RETVAL(double) FNAME(FPARAMS(double)) MOD SUFF                \
  DOC PREF RETVAL(uint) FNAME(FPARAMS(uint)) MOD SUFF     \
  DOC PREF RETVAL(uint8_t) FNAME(FPARAMS(uint8_t)) MOD SUFF
#define FORPY_GEN_TMPLMARK_AT template<typename AT>
#define FORPY_INSERT_TPARMS_AT(FUNC) FUNC(AT)
#define FORPY_GEN_TMPLINST_AT(AT) <AT>
// Only input type (regression).
#define FORPY_GEN_FUNCHEAD_ITR(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF) \
  DOC PREF RETVAL(float) FNAME(FPARAMS(float)) MOD SUFF                 \
  DOC PREF RETVAL(double) FNAME(FPARAMS(double)) MOD SUFF
#define FORPY_GEN_TMPLMARK_ITR template<typename IT>
#define FORPY_INSERT_TPARMS_ITR(FUNC) FUNC(IT)
#define FORPY_GEN_TMPLINST_ITR(IT) <IT>
// Input and annotation type.
#define FORPY_GEN_FUNCHEAD_ITAT(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF) \
  DOC PREF RETVAL(float, float) FNAME(FPARAMS(float, float)) MOD SUFF         \
  DOC PREF RETVAL(float, double) FNAME(FPARAMS(float, double)) MOD SUFF \
  DOC PREF RETVAL(float, uint8_t) FNAME(FPARAMS(float, uint8_t)) MOD SUFF \
  DOC PREF RETVAL(float, uint) FNAME(FPARAMS(float, uint)) MOD SUFF \
    DOC PREF RETVAL(double, float) FNAME(FPARAMS(double, float)) MOD SUFF \
    DOC PREF RETVAL(double, double) FNAME(FPARAMS(double, double)) MOD SUFF \
    DOC PREF RETVAL(double, uint8_t) FNAME(FPARAMS(double, uint8_t)) MOD SUFF \
    DOC PREF RETVAL(double, uint) FNAME(FPARAMS(double, uint)) MOD SUFF   \
  DOC PREF RETVAL(uint8_t, float) FNAME(FPARAMS(uint8_t, float)) MOD SUFF   \
  DOC PREF RETVAL(uint8_t, double) FNAME(FPARAMS(uint8_t, double)) MOD SUFF \
  DOC PREF RETVAL(uint8_t, uint8_t) FNAME(FPARAMS(uint8_t, uint8_t)) MOD SUFF \
    DOC PREF RETVAL(uint8_t, uint) FNAME(FPARAMS(uint8_t, uint)) MOD SUFF   \
    DOC PREF RETVAL(uint, float) FNAME(FPARAMS(uint, float)) MOD SUFF \
    DOC PREF RETVAL(uint, double) FNAME(FPARAMS(uint, double)) MOD SUFF \
    DOC PREF RETVAL(uint, uint8_t) FNAME(FPARAMS(uint, uint8_t)) MOD SUFF \
    DOC PREF RETVAL(uint, uint) FNAME(FPARAMS(uint, uint)) MOD SUFF
#define FORPY_GEN_TMPLMARK_ITAT template<typename IT, typename AT>
#define FORPY_INSERT_TPARMS_ITAT(FUNC) FUNC(IT, AT)
#define FORPY_GEN_TMPLINST_ITAT(IT, AT) <IT, AT>
// Input and feature type.
#define FORPY_GEN_FUNCHEAD_ITFT(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF) \
  DOC PREF RETVAL(float, float) FNAME(FPARAMS(float, float)) MOD SUFF   \
  DOC PREF RETVAL(double, double) FNAME(FPARAMS(double, double)) MOD SUFF \
  DOC PREF RETVAL(uint8_t, uint8_t) FNAME(FPARAMS(uint8_t, uint8_t)) MOD SUFF \
  DOC PREF RETVAL(uint, uint) FNAME(FPARAMS(uint, uint)) MOD SUFF
#define FORPY_GEN_TMPLMARK_ITFT template<typename IT, typename FT>
#define FORPY_INSERT_TPARMS_ITFT(FUNC) FUNC(IT, FT)
#define FORPY_GEN_TMPLINST_ITFT(IT, FT) <IT, FT>
// Input and feature type, equal.
#define FORPY_GEN_FUNCHEAD_ITFTEQ(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF) \
  DOC PREF RETVAL(float, float) FNAME(FPARAMS(float, float)) MOD SUFF   \
  DOC PREF RETVAL(double, double) FNAME(FPARAMS(double, double)) MOD SUFF \
  DOC PREF RETVAL(uint, uint) FNAME(FPARAMS(uint, uint)) MOD SUFF         \
  DOC PREF RETVAL(uint8_t, uint8_t) FNAME(FPARAMS(uint8_t, uint8_t)) MOD SUFF
#define FORPY_GEN_TMPLMARK_ITFTEQ template<typename IT, typename FT>
#define FORPY_INSERT_TPARMS_ITFTEQ(FUNC) FUNC(IT, FT)
#define FORPY_GEN_TMPLINST_ITFTEQ(IT, FT) <IT, FT>
// Input, feature and annotation type.
#define FORPY_GEN_FUNCHEAD_ITFTAT(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF)     \
  DOC PREF RETVAL(float, float, float) FNAME(FPARAMS(float, float, float)) MOD SUFF \
    DOC PREF RETVAL(float, float, double) FNAME(FPARAMS(float, float, double)) MOD SUFF \
    DOC PREF RETVAL(float, float, uint8_t) FNAME(FPARAMS(float, float, uint8_t)) MOD SUFF \
    DOC PREF RETVAL(float, float, uint) FNAME(FPARAMS(float, float, uint)) MOD SUFF \
    DOC PREF RETVAL(double, double, float) FNAME(FPARAMS(double, double, float)) MOD SUFF \
    DOC PREF RETVAL(double, double, double) FNAME(FPARAMS(double, double, double)) MOD SUFF \
    DOC PREF RETVAL(double, double, uint8_t) FNAME(FPARAMS(double, double, uint8_t)) MOD SUFF \
    DOC PREF RETVAL(double, double, uint) FNAME(FPARAMS(double, double, uint)) MOD SUFF \
    DOC PREF RETVAL(uint8_t, uint8_t, float) FNAME(FPARAMS(uint8_t, uint8_t, float)) MOD SUFF \
    DOC PREF RETVAL(uint8_t, uint8_t, double) FNAME(FPARAMS(uint8_t, uint8_t, double)) MOD SUFF \
  DOC PREF RETVAL(uint8_t, uint8_t, uint8_t) FNAME(FPARAMS(uint8_t, uint8_t, uint8_t)) MOD SUFF \
  DOC PREF RETVAL(uint8_t, uint8_t, uint) FNAME(FPARAMS(uint8_t, uint8_t, uint)) MOD SUFF \
    DOC PREF RETVAL(uint, uint, float) FNAME(FPARAMS(uint, uint, float)) MOD SUFF \
    DOC PREF RETVAL(uint, uint, double) FNAME(FPARAMS(uint, uint, double)) MOD SUFF \
  DOC PREF RETVAL(uint, uint, uint8_t) FNAME(FPARAMS(uint, uint, uint8_t)) MOD SUFF \
  DOC PREF RETVAL(uint, uint, uint) FNAME(FPARAMS(uint, uint, uint)) MOD SUFF
#define FORPY_GEN_TMPLMARK_ITFTAT template<typename IT, typename FT, typename AT>
#define FORPY_INSERT_TPARMS_ITFTAT(FUNC) FUNC(IT, FT, AT)
#define FORPY_GEN_TMPLINST_ITFTAT(IT, FT, AT) <IT, FT, AT>
// Input, feature and annotation type, regression valid types only.
#define FORPY_GEN_FUNCHEAD_ITFTATR(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF)     \
  DOC PREF RETVAL(float, float, float) FNAME(FPARAMS(float, float, float)) MOD SUFF \
  DOC PREF RETVAL(double, double, double) FNAME(FPARAMS(double, double, double)) MOD SUFF
#define FORPY_GEN_TMPLMARK_ITFTATR template<typename IT, typename FT, typename AT>
#define FORPY_INSERT_TPARMS_ITFTATR(FUNC) FUNC(IT, FT, AT)
#define FORPY_GEN_TMPLINST_ITFTATR(IT, FT, AT) <IT, FT, AT>

/***************************** Functions **************************************/

#define FORPY_EMPTY(...) 

#define FORPY_GEN_CALLER(CALLF, DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF) \
    CALLF(DOC, PREF, RETVAL, FNAME, FPARAMS, MOD, SUFF)
    
#define FORPY_DECL(FID, TMPL, PREF, SUFF) \
    FORPY_GEN_CALLER(FORPY_GEN_FUNCHEAD_##TMPL, \
                     FID##_DOC, \
                     PREF, \
                     FID##_RET, \
                     FID##_NAME, \
                     FID##_PARAMTYPESNNAMESNDEF, \
                     FID##_MOD, \
                     SUFF)

#define FORPY_CALL(FID, TMPL)                   \
  FORPY_GEN_CALLER(FORPY_GEN_FUNCHEAD_##TMPL,   \
                   ,                            \
                   ,                            \
                   FORPY_EMPTY,                 \
                   FID##_NAME,                  \
                   FID##_PARAMCALL,             \
                   ,                            \
                   ;)

#define FORPY_NOTAVAIL_SUFF_(FUNC) \
  { throw Forpy_Exception("The function " #FUNC " is not implemented for this type!"); };


#define FORPY_NOTAVAIL_SUFF(FUNC) FORPY_NOTAVAIL_SUFF_(FUNC)

#define FORPY_IMPL_NOTAVAIL(FID, TMPL, QUAL) \
  FORPY_GEN_CALLER(FORPY_GEN_FUNCHEAD_##TMPL, \
                   FID##_DOC, \
                   ,          \
                   FID##_RET, \
                   QUAL::FID##_NAME, \
                   FID##_PARAMTYPESNNAMES, \
                   FID##_MOD, \
                   FORPY_NOTAVAIL_SUFF(QUAL::FID##_NAME))

#define FORPY_IMPL_DIRECT(FID, PREF, TMPL, QUAL, SUFF) \
  FORPY_GEN_CALLER(FORPY_GEN_FUNCHEAD_##TMPL,           \
                   FID##_DOC,                   \
                   PREF,                        \
                   FID##_RET,                   \
                   QUAL::FID##_NAME,            \
                   FID##_PARAMTYPESNNAMES,      \
                   FID##_MOD,                   \
                   SUFF)

// Stages for DECLARE_IMPL.
#define FORPY_DECL_IMPL__(TMPLFUNC, PREF, RET, FNAME, FPARAMS, MOD, SUFF) \
  FORPY_UNPACK(TMPLFUNC) PREF FORPY_UNPACK(FORPY_UNPACK RET) FNAME##_impl FPARAMS MOD
#define FORPY_DECL_IMPL_(TMPLFUNC, PREF, RET, FNAME, FPARAMS, MOD, SUFF) \
  FORPY_DECL_IMPL__(FORPY_GEN_TMPLMARK_##TMPLFUNC, PREF, (RET), FNAME, FPARAMS, MOD, SUFF)
#define FORPY_DECL_IMPL(FID, TMPL, SUFF)     \
  FORPY_DECL_IMPL_(TMPL,\
                   ,                                                   \
                   FORPY_INSERT_TPARMS_##TMPL(FID##_RET),               \
                   FID##_NAME,                                          \
                   (FORPY_INSERT_TPARMS_##TMPL(FID##_PARAMTYPESNNAMESNDEF)), \
                   FID##_MOD,                                           \
                   SUFF)

// Stages for PROXY.
#define FORPY_GEN_FORWARD(FNAME, PFUNC) \
  { return FNAME##_impl PFUNC; };
#define FORPY_PROXY_(FNAME, PFUNC) \
  FORPY_GEN_FORWARD(FNAME, PFUNC)
#define FORPY_PROXY(FID, TMPL, QUAL) \
  FORPY_GEN_CALLER(FORPY_GEN_FUNCHEAD_##TMPL, \
                   FID##_DOC, \
                   , \
                   FID##_RET, \
                   QUAL::FID##_NAME, \
                   FID##_PARAMTYPESNNAMES, \
                   FID##_MOD, \
                   FORPY_PROXY_(FID##_NAME, (FID##_PARAMNAMES)))
#define FORPY_IMPL_HEAD(FID, TMPL, QUAL)        \
  FORPY_DECL_IMPL_(TMPL, \
                   ,                                                   \
                   FORPY_INSERT_TPARMS_##TMPL(FID##_RET),               \
                   QUAL::FID##_NAME,                                    \
                   (FORPY_INSERT_TPARMS_##TMPL(FID##_PARAMTYPESNNAMES)), \
                   FID##_MOD,                                           \
                   )
#define FORPY_IMPL(FID, TMPL, QUAL)        \
  FORPY_PROXY(FID, TMPL, QUAL) \
  FORPY_IMPL_HEAD(FID, TMPL, QUAL)
