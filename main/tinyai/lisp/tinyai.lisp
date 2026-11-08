;;;; tinyai.lisp -- Common Lisp front end for tinyai
;;;;
;;;; tinyai is a small neural network whose number crunching lives in C and in
;;;; hand-written x86-64 assembly (AVX2 + FMA).  This file is the Lisp side: it
;;;; loads build/tinyai.dll through SBCL's built-in foreign-function interface
;;;; (sb-alien) and wraps it in ordinary Lisp functions, so you can build, train
;;;; and poke at networks from a REPL while the heavy lifting still runs in
;;;; assembly.
;;;;
;;;;   $ sbcl
;;;;   * (load "lisp/tinyai.lisp")
;;;;   * (in-package :tinyai)
;;;;   * (with-data ((train (make-spirals :seed 1234))
;;;;                 (test  (make-spirals :seed 1235)))
;;;;       (with-net ((net (make-net '(2 64 64 3) :seed 1236)))
;;;;         (fit net train :epochs 100 :test test)
;;;;         (plot-regions net test)))
;;;;
;;;;   * (dot '(1 2 3 4) '(5 6 7 8))      ; the assembly kernel, called from Lisp
;;;;   70.0
;;;;
;;;; Build the DLL first with  .\build.ps1  (it is written to build/tinyai.dll).

(defpackage #:tinyai
  (:use #:common-lisp)
  (:export #:tinyai-error
           ;; the machine and the assembly kernels
           #:avx2-p #:kernel-name #:dot #:axpy #:dot-reference
           ;; datasets
           #:make-spirals #:free-data #:with-data
           #:data-size #:data-dim #:data-classes #:data-point #:data-label
           ;; networks
           #:make-net #:free-net #:with-net
           #:net-dims #:net-params #:net-backend
           #:train-epoch #:evaluate #:predict #:logits
           ;; training and display
           #:cosine-lr #:fit #:plot-regions))

(in-package #:tinyai)

;;; ---------------------------------------------------------------------------
;;; Errors and loading the library
;;; ---------------------------------------------------------------------------

(define-condition tinyai-error (error)
  ((message :initarg :message :reader tinyai-error-message))
  (:report (lambda (condition stream)
             (format stream "tinyai: ~a" (tinyai-error-message condition)))))

(defun fail (control &rest arguments)
  (error 'tinyai-error :message (apply #'format nil control arguments)))

(eval-when (:compile-toplevel :load-toplevel :execute)

  (defparameter *dll-path*
    (let ((here (or *load-truename* *compile-file-truename*)))
      (and here
           (merge-pathnames (make-pathname :directory '(:relative :up "build")
                                           :name "tinyai" :type "dll")
                            here)))
    "Where tinyai.dll is expected: ../build/tinyai.dll relative to this file.")

  (defvar *library-loaded* nil)

  (defun load-library (&optional (path *dll-path*))
    "Load tinyai.dll into this Lisp image (once)."
    (unless *library-loaded*
      (let ((dll (and path (probe-file path))))
        (unless dll
          (fail "~a not found - build it first with  .\\build.ps1" (or path "tinyai.dll")))
        (sb-alien:load-shared-object dll :dont-save t)
        (setf *library-loaded* t)))
    t)

  ;; Must run before the alien routines below are compiled, or SBCL would warn
  ;; about undefined foreign symbols.
  (load-library))

;;; ---------------------------------------------------------------------------
;;; The foreign routines (flat C API, see src/tinyai_api.h)
;;; ---------------------------------------------------------------------------

(defmacro with-c-float-environment (&body body)
  "Run BODY with floating-point traps masked.  SBCL traps overflow / invalid /
divide-by-zero by default; the C code is entitled to see IEEE infinities and
NaNs without Lisp turning them into a condition in the middle of a foreign call."
  `(sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero :inexact)
     ,@body))

(deftype float-vector () '(simple-array single-float (*)))

(sb-alien:define-alien-routine ("tai_cpu_has_avx2_fma" %avx2-p) (sb-alien:unsigned 32))
(sb-alien:define-alien-routine ("tai_kernel_name" %kernel-name) sb-alien:c-string)

(sb-alien:define-alien-routine ("tai_dot" %dot) single-float
  (a sb-sys:system-area-pointer) (b sb-sys:system-area-pointer) (n (sb-alien:unsigned 64)))
(sb-alien:define-alien-routine ("tai_dot_scalar" %dot-scalar) single-float
  (a sb-sys:system-area-pointer) (b sb-sys:system-area-pointer) (n (sb-alien:unsigned 64)))
(sb-alien:define-alien-routine ("tai_axpy" %axpy) sb-alien:void
  (n (sb-alien:unsigned 64)) (alpha single-float)
  (x sb-sys:system-area-pointer) (y sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_axpy_scalar" %axpy-scalar) sb-alien:void
  (n (sb-alien:unsigned 64)) (alpha single-float)
  (x sb-sys:system-area-pointer) (y sb-sys:system-area-pointer))

(sb-alien:define-alien-routine ("tai_spirals_new" %spirals-new) sb-sys:system-area-pointer
  (per-class (sb-alien:unsigned 64)) (classes (sb-alien:unsigned 64))
  (turns single-float) (noise single-float) (seed (sb-alien:unsigned 64)))
(sb-alien:define-alien-routine ("tai_data_free" %data-free) sb-alien:void
  (d sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_data_size" %data-size) (sb-alien:unsigned 64)
  (d sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_data_dim" %data-dim) (sb-alien:unsigned 64)
  (d sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_data_classes" %data-classes) (sb-alien:unsigned 64)
  (d sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_data_x" %data-x) sb-sys:system-area-pointer
  (d sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_data_y" %data-y) sb-sys:system-area-pointer
  (d sb-sys:system-area-pointer))

(sb-alien:define-alien-routine ("tai_net_new" %net-new) sb-sys:system-area-pointer
  (dims sb-sys:system-area-pointer) (n-layers (sb-alien:unsigned 64))
  (seed (sb-alien:unsigned 64)) (force-scalar (sb-alien:signed 32)))
(sb-alien:define-alien-routine ("tai_net_free" %net-free) sb-alien:void
  (n sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_net_n_params" %net-n-params) (sb-alien:unsigned 64)
  (n sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_net_inputs" %net-inputs) (sb-alien:unsigned 64)
  (n sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_net_outputs" %net-outputs) (sb-alien:unsigned 64)
  (n sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_net_backend" %net-backend) sb-alien:c-string
  (n sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_net_train_epoch" %net-train-epoch) double-float
  (n sb-sys:system-area-pointer) (d sb-sys:system-area-pointer)
  (lr single-float) (batch (sb-alien:unsigned 64)))
(sb-alien:define-alien-routine ("tai_net_evaluate" %net-evaluate) sb-alien:void
  (n sb-sys:system-area-pointer) (d sb-sys:system-area-pointer)
  (loss sb-sys:system-area-pointer) (acc sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_net_predict" %net-predict) (sb-alien:unsigned 64)
  (n sb-sys:system-area-pointer) (x sb-sys:system-area-pointer))
(sb-alien:define-alien-routine ("tai_net_logits" %net-logits) sb-sys:system-area-pointer
  (n sb-sys:system-area-pointer) (x sb-sys:system-area-pointer))

;;; ---------------------------------------------------------------------------
;;; The machine, and the assembly kernels on Lisp vectors
;;; ---------------------------------------------------------------------------

(defun avx2-p ()
  "True if this CPU *and* operating system support AVX2+FMA, i.e. the assembly
kernels (rather than the plain-C fallback) are used."
  (= 1 (%avx2-p)))

(defun kernel-name ()
  "Name of the kernel set DOT and AXPY use."
  (%kernel-name))

(defun as-float-vector (sequence)
  "SEQUENCE as a (simple-array single-float (*)).  A vector that already has that
type is returned as is, without copying."
  (if (typep sequence 'float-vector)
      sequence
      (map 'float-vector (lambda (v) (float v 1f0)) sequence)))

(defun dot (a b &key scalar)
  "Dot product of two equal-length numeric sequences, computed by the assembly
kernel (AVX2+FMA).  Single-float vectors are passed to C without copying.
:SCALAR T runs the plain-C reference implementation instead."
  (let ((a (as-float-vector a))
        (b (as-float-vector b)))
    (unless (= (length a) (length b))
      (fail "dot: the vectors have different lengths (~d and ~d)" (length a) (length b)))
    (sb-sys:with-pinned-objects (a b)
      (with-c-float-environment
        (funcall (if scalar #'%dot-scalar #'%dot)
                 (sb-sys:vector-sap a) (sb-sys:vector-sap b) (length a))))))

(defun axpy (alpha x y &key scalar)
  "Destructively compute  y <- y + alpha * x  with the assembly kernel and return Y.
Y must be a (simple-array single-float (*)); X may be any numeric sequence of the
same length."
  (check-type y float-vector)
  (let ((x (as-float-vector x)))
    (unless (= (length x) (length y))
      (fail "axpy: the vectors have different lengths (~d and ~d)" (length x) (length y)))
    (sb-sys:with-pinned-objects (x y)
      (with-c-float-environment
        (funcall (if scalar #'%axpy-scalar #'%axpy)
                 (length y) (float alpha 1f0) (sb-sys:vector-sap x) (sb-sys:vector-sap y))))
    y))

(defun dot-reference (a b)
  "The same dot product in pure Lisp with double-float accumulation: what the
kernels are tested against."
  (let ((sum 0d0))
    (map nil (lambda (u v) (incf sum (* (float u 1d0) (float v 1d0)))) a b)
    sum))

;;; ---------------------------------------------------------------------------
;;; Handles: C owns the memory, Lisp owns the lifetime
;;; ---------------------------------------------------------------------------

(defun register-finalizer (object address free-function)
  "Safety net for REPL use: free the C object if OBJECT is garbage-collected
without an explicit FREE-DATA / FREE-NET.  The closure holds only the integer
ADDRESS, never OBJECT itself."
  (sb-ext:finalize object
                   (lambda () (funcall free-function (sb-sys:int-sap address)))
                   :dont-save t))

(defun null-sap-p (sap)
  (zerop (sb-sys:sap-int sap)))

;;; ---------------------------------------------------------------------------
;;; Datasets
;;; ---------------------------------------------------------------------------

(defstruct (data (:constructor %make-data (sap size dim classes)) (:copier nil))
  (sap nil)
  (size 0 :type fixnum)
  (dim 0 :type fixnum)
  (classes 0 :type fixnum))

(defmethod print-object ((d data) stream)
  (print-unreadable-object (d stream :type t)
    (if (data-sap d)
        (format stream "~d points, ~d features, ~d classes"
                (data-size d) (data-dim d) (data-classes d))
        (write-string "(freed)" stream))))

(defun data-handle (d)
  (check-type d data)
  (or (data-sap d) (fail "this dataset has already been freed")))

(defun make-spirals (&key (per-class 200) (classes 3) (turns 1.0) (noise 0.4) (seed 1234))
  "A fresh spiral classification dataset: CLASSES interleaved arms, PER-CLASS
points each, TURNS revolutions per arm, angular NOISE in radians (the arms are
2*pi/CLASSES apart).  The same arguments always give the same points."
  (check-type per-class (integer 1 16777216))
  (check-type classes (integer 1 64))
  (check-type turns real)
  (check-type noise real)
  (check-type seed (unsigned-byte 62))
  (let ((sap (with-c-float-environment
               (%spirals-new per-class classes (float turns 1f0) (float noise 1f0) seed))))
    (when (null-sap-p sap)
      (fail "could not create the dataset (bad arguments or out of memory)"))
    (let ((d (%make-data sap (%data-size sap) (%data-dim sap) (%data-classes sap))))
      (register-finalizer d (sb-sys:sap-int sap) #'%data-free)
      d)))

(defun free-data (d)
  "Release the C memory behind D.  Safe to call more than once."
  (check-type d data)
  (when (data-sap d)
    (sb-ext:cancel-finalization d)
    (%data-free (data-sap d))
    (setf (data-sap d) nil))
  nil)

(defmacro with-data (bindings &body body)
  "Like LET*, but every value is a dataset that is freed when BODY is left."
  (if (null bindings)
      `(progn ,@body)
      (destructuring-bind ((var form) &rest more) bindings
        `(let ((,var ,form))
           (unwind-protect (with-data ,more ,@body)
             (free-data ,var))))))

(defun data-point (d i)
  "Features of sample I as a list of single-floats."
  (unless (< -1 i (data-size d))
    (fail "sample index ~d is outside 0..~d" i (1- (data-size d))))
  (let ((x (%data-x (data-handle d))))
    (loop for k below (data-dim d)
          collect (sb-sys:sap-ref-single x (* 4 (+ (* i (data-dim d)) k))))))

(defun data-label (d i)
  "Class (0-based) of sample I."
  (unless (< -1 i (data-size d))
    (fail "sample index ~d is outside 0..~d" i (1- (data-size d))))
  (sb-sys:sap-ref-32 (%data-y (data-handle d)) (* 4 i)))

;;; ---------------------------------------------------------------------------
;;; Networks
;;; ---------------------------------------------------------------------------

(defstruct (net (:constructor %make-net (sap dims params backend inputs outputs))
                (:copier nil))
  (sap nil)
  (dims nil :read-only t)
  (params 0 :type fixnum :read-only t)
  (backend "" :read-only t)
  (inputs 0 :type fixnum :read-only t)
  (outputs 0 :type fixnum :read-only t))

(defmethod print-object ((n net) stream)
  (print-unreadable-object (n stream :type t)
    (if (net-sap n)
        (format stream "~{~d~^-~}, ~:d parameters, ~a" (net-dims n) (net-params n) (net-backend n))
        (write-string "(freed)" stream))))

(defun net-handle (n)
  (check-type n net)
  (or (net-sap n) (fail "this network has already been freed")))

(defun make-net (dims &key (seed 1) force-scalar)
  "A new fully-connected network.  DIMS lists the layer widths, e.g. (2 64 64 3):
2 inputs, two hidden ReLU layers of 64 neurons, 3 output classes.  SEED fixes the
initial weights and the shuffling order.  FORCE-SCALAR T uses the plain-C kernels
even when the CPU supports the assembly ones."
  (unless (and (listp dims)
               (<= 2 (length dims) 9)
               (every (lambda (w) (typep w '(integer 1 16777216))) dims))
    (fail "DIMS must be a list of 2 to 9 positive integers such as (2 64 64 3), not ~s" dims))
  (check-type seed (unsigned-byte 62))
  (let* ((widths (make-array (length dims) :element-type '(unsigned-byte 64)
                                           :initial-contents dims))
         (sap (sb-sys:with-pinned-objects (widths)
                (with-c-float-environment
                  (%net-new (sb-sys:vector-sap widths) (1- (length dims)) seed
                            (if force-scalar 1 0))))))
    (when (null-sap-p sap)
      (fail "could not create the network (out of memory?)"))
    (let ((n (%make-net sap (copy-list dims) (%net-n-params sap) (%net-backend sap)
                        (%net-inputs sap) (%net-outputs sap))))
      (register-finalizer n (sb-sys:sap-int sap) #'%net-free)
      n)))

(defun free-net (n)
  "Release the C memory behind N.  Safe to call more than once."
  (check-type n net)
  (when (net-sap n)
    (sb-ext:cancel-finalization n)
    (%net-free (net-sap n))
    (setf (net-sap n) nil))
  nil)

(defmacro with-net (bindings &body body)
  "Like LET*, but every value is a network that is freed when BODY is left."
  (if (null bindings)
      `(progn ,@body)
      (destructuring-bind ((var form) &rest more) bindings
        `(let ((,var ,form))
           (unwind-protect (with-net ,more ,@body)
             (free-net ,var))))))

(defun check-fit (net data)
  "A dataset fits a network if the feature counts agree and no label can exceed
the number of outputs."
  (unless (= (data-dim data) (net-inputs net))
    (fail "the dataset has ~d features per sample but the network takes ~d"
          (data-dim data) (net-inputs net)))
  (unless (<= (data-classes data) (net-outputs net))
    (fail "the dataset has ~d classes but the network has only ~d outputs"
          (data-classes data) (net-outputs net))))

(defun train-epoch (net data &key (lr 0.02) (batch 1))
  "One shuffled pass of stochastic gradient descent over DATA.  Weights change
every BATCH samples.  Returns the mean training loss seen during the pass."
  (check-type lr (real (0)))
  (check-type batch (integer 1))
  (check-fit net data)
  (let ((loss (with-c-float-environment
                (%net-train-epoch (net-handle net) (data-handle data) (float lr 1f0) batch))))
    (when (/= loss loss)
      (fail "training returned NaN"))
    loss))

(defun evaluate (net data)
  "Returns two values: the mean cross-entropy loss and the accuracy in percent of
NET on DATA (no learning)."
  (check-fit net data)
  (sb-alien:with-alien ((loss sb-alien:double) (accuracy sb-alien:double))
    (with-c-float-environment
      (%net-evaluate (net-handle net) (data-handle data)
                     (sb-alien:alien-sap (sb-alien:addr loss))
                     (sb-alien:alien-sap (sb-alien:addr accuracy))))
    (values loss accuracy)))

(defun input-vector (net x)
  (let ((v (as-float-vector x)))
    (unless (= (length v) (net-inputs net))
      (fail "the network takes ~d inputs, not ~d" (net-inputs net) (length v)))
    v))

(defun predict (net x)
  "The class (0-based) NET assigns to the input X, a sequence of numbers."
  (let ((v (input-vector net x)))
    (sb-sys:with-pinned-objects (v)
      (with-c-float-environment
        (%net-predict (net-handle net) (sb-sys:vector-sap v))))))

(defun logits (net x)
  "The raw output scores of NET for the input X, as a fresh vector of single-floats."
  (let* ((v (input-vector net x))
         (out (make-array (net-outputs net) :element-type 'single-float)))
    (sb-sys:with-pinned-objects (v)
      (let ((sap (with-c-float-environment
                   (%net-logits (net-handle net) (sb-sys:vector-sap v)))))
        (dotimes (i (length out))
          (setf (aref out i) (sb-sys:sap-ref-single sap (* 4 i))))))
    out))

;;; ---------------------------------------------------------------------------
;;; Training loop and display
;;; ---------------------------------------------------------------------------

(defun cosine-lr (lr lr-final epoch epochs)
  "Cosine learning-rate decay from LR (epoch 1) towards LR*LR-FINAL.  Same
arithmetic, in the same order and precision, as ai_cosine_lr() in src/train.c, so
a run driven from Lisp is numerically identical to the C program's."
  (let* ((lr (float lr 1f0))
         (lr-final (float lr-final 1f0))
         (progress (/ (float (1- epoch) 1d0) (float epochs 1d0)))
         (c (float (cos (* pi progress)) 1f0)))
    (* lr (+ lr-final (* (- 1f0 lr-final) 0.5f0 (+ 1f0 c))))))

(defun fit (net train &key (epochs 300) (lr 0.02) (lr-final 0.05) (batch 1)
                           test (log-every 25) (stream *standard-output*))
  "Train NET on TRAIN for EPOCHS passes with cosine learning-rate decay, printing
a progress table to STREAM (NIL = silent) every LOG-EVERY epochs.  If TEST is
given its accuracy is reported too.  Returns a property list with :TRAIN-LOSS
:TRAIN-ACCURACY :TEST-LOSS :TEST-ACCURACY (NIL without TEST) :SECONDS."
  (check-type epochs (integer 1))
  (check-type log-every (integer 1))
  (when stream
    (format stream " epoch        lr       loss    train%     test%~%"))
  (let ((seconds 0d0))
    (loop for epoch from 1 to epochs
          for lr-now = (cosine-lr lr lr-final epoch epochs)
          do (let ((start (get-internal-real-time)))
               (train-epoch net train :lr lr-now :batch batch)
               (incf seconds (/ (- (get-internal-real-time) start)
                                internal-time-units-per-second)))
             (when (and stream (or (= epoch 1) (zerop (mod epoch log-every)) (= epoch epochs)))
               (multiple-value-bind (train-loss train-acc) (evaluate net train)
                 (format stream "~6d  ~8,5f  ~9,5f  ~7,2f%  ~a~%"
                         epoch (float lr-now 1d0) train-loss train-acc
                         (if test
                             (format nil "~7,2f%" (nth-value 1 (evaluate net test)))
                             "      -")))))
    (multiple-value-bind (train-loss train-acc) (evaluate net train)
      (multiple-value-bind (test-loss test-acc) (if test (evaluate net test) (values nil nil))
        (list :train-loss train-loss :train-accuracy train-acc
              :test-loss test-loss :test-accuracy test-acc
              :seconds seconds)))))

(defun plot-regions (net data &key (width 66) (height 33) (limit 1.15)
                                   (stream *standard-output*))
  "ASCII picture of what a 2-input NET has learned: the plane is shaded by the
predicted class, then the points of DATA are drawn on top (a digit = the point's
true class, ! = a point the network gets wrong).  The same picture, cell for
cell, as the C program prints."
  (unless (= (net-inputs net) 2)
    (fail "plot-regions needs a network with 2 inputs, this one has ~d" (net-inputs net)))
  (check-fit net data)
  (let* ((region " .:-=+*#")
         (lim (float limit 1f0))
         (w (float width 1f0))
         (h (float height 1f0))
         (grid (make-array (list height width) :element-type 'character))
         (p (make-array 2 :element-type 'single-float))
         (handle (net-handle net))
         (xs (%data-x (data-handle data))))
    (flet ((classify ()
             (sb-sys:with-pinned-objects (p)
               (%net-predict handle (sb-sys:vector-sap p)))))
      (with-c-float-environment
        (dotimes (r height)
          (dotimes (c width)
            (setf (aref p 0) (+ (- lim) (/ (* (* 2f0 lim) (+ (float c 1f0) 0.5f0)) w))
                  (aref p 1) (- lim (/ (* (* 2f0 lim) (+ (float r 1f0) 0.5f0)) h)))
            (setf (aref grid r c) (char region (mod (classify) (length region))))))
        (dotimes (i (data-size data))
          (setf (aref p 0) (sb-sys:sap-ref-single xs (* 4 (* 2 i)))
                (aref p 1) (sb-sys:sap-ref-single xs (* 4 (1+ (* 2 i)))))
          (let ((c (truncate (* (/ (+ (aref p 0) lim) (* 2f0 lim)) w)))
                (r (truncate (* (/ (- lim (aref p 1)) (* 2f0 lim)) h))))
            (when (and (< -1 c width) (< -1 r height))
              (setf (aref grid r c)
                    (if (= (classify) (data-label data i))
                        (code-char (+ (char-code #\0) (mod (data-label data i) 10)))
                        #\!)))))))
    (when stream
      (let ((border (concatenate 'string "+" (make-string width :initial-element #\-) "+")))
        (format stream "~%What the network learned (shading = predicted class, digits = test points):~%")
        (write-line border stream)
        (dotimes (r height)
          (format stream "|~a|~%" (coerce (loop for c below width collect (aref grid r c)) 'string)))
        (write-line border stream)
        (format stream "class 0 = blank   class 1 = '.'   class 2 = ':'     '!' = misclassified test point~%")))
    grid))
