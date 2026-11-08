;;;; test.lisp -- tests for the Lisp front end
;;;;
;;;;   sbcl --noinform --non-interactive --no-sysinit --no-userinit --load lisp\test.lisp
;;;;
;;;; Exit code 0 = everything passed.  The headline test trains the spiral network
;;;; from Lisp and checks that the result agrees with what build\tinyai.exe prints
;;;; for the same settings: same final accuracies and loss, and the same ASCII
;;;; picture cell for cell.  That exercises the whole bridge at once: argument
;;;; marshalling (including Win64's positional float slots), seeds, the learning-
;;;; rate schedule, the shared C training loop, and the assembly underneath.

(load (merge-pathnames "tinyai.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:tinyai)

(defvar *checks* 0)
(defvar *failures* 0)

(defmacro check (form &optional (control "") &rest arguments)
  `(progn
     (incf *checks*)
     (unless ,form
       (incf *failures*)
       (format t "~&    FAIL: ~s  ~?~%" ',form ,control (list ,@arguments)))))

(defmacro signals-tinyai-error (form)
  "True if FORM signals TINYAI-ERROR."
  `(handler-case (progn ,form nil)
     (tinyai-error () t)))

(defmacro section (title &body body)
  `(progn
     (format t "  ~a ... " ,title)
     (finish-output)
     (let ((before *failures*))
       (handler-case (progn ,@body)
         (error (e)
           (incf *failures*)
           (format t "~&    ERROR: ~a~%" e)))
       (format t "~a~%" (if (= before *failures*) "ok" "FAILED")))))

(defvar *rng* (sb-ext:seed-random-state 20260609))

(defun random-floats (n)
  (let ((v (make-array n :element-type 'single-float)))
    (dotimes (i n v)
      (setf (aref v i) (- (random 2f0 *rng*) 1f0)))))

(defparameter *sizes*
  (append (loop for i from 0 to 70 collect i) '(95 96 97 127 128 129 255 256 257 1003 4099)))

;;; ---------------------------------------------------------------------------

(format t "tinyai Lisp tests -- SBCL ~a, kernels: ~a~%" (lisp-implementation-version) (kernel-name))

(section "dot / axpy agree with a pure-Lisp double-precision reference"
  (dolist (scalar (if (avx2-p) '(nil t) '(t)))
    (dolist (n *sizes*)
      (let* ((a (random-floats n))
             (b (random-floats n))
             (abs-sum (loop for u across a for v across b sum (abs (* (float u 1d0) v))))
             (want (dot-reference a b))
             (got (dot a b :scalar scalar)))
        (check (<= (abs (- got want)) (+ (* 1d-5 abs-sum) 1d-6))
               "dot n=~d scalar=~a got ~f want ~f" n scalar got want))
      (let* ((x (random-floats n))
             (y (random-floats n))
             (y0 (copy-seq y))
             (x0 (copy-seq x))
             (alpha 0.75f0))
        (axpy alpha x y :scalar scalar)
        (check (equalp x x0) "axpy modified x (n=~d)" n)
        (check (loop for i below n
                     always (let ((want (+ (float (aref y0 i) 1d0) (* alpha (float (aref x i) 1d0)))))
                              (<= (abs (- (aref y i) want))
                                  (+ (* 1d-6 (+ (abs (aref y0 i)) (abs (* alpha (aref x i))))) 1d-30))))
               "axpy wrong result (n=~d scalar=~a)" n scalar)))))

(section "dot / axpy take lists and mixed numbers, and reject bad input"
  (check (= (dot '(1 2 3) '(4 5 6)) 32.0))
  (check (= (dot '(1/2 0.5d0 2) #(2 2 1)) 4.0))
  (check (= (dot '() '()) 0.0))
  (check (equalp (axpy 2 '(1 2 3) (make-array 3 :element-type 'single-float :initial-element 1f0))
                 (make-array 3 :element-type 'single-float :initial-contents '(3f0 5f0 7f0))))
  (check (signals-tinyai-error (dot '(1 2 3) '(1 2))))
  (check (signals-tinyai-error (axpy 1 '(1 2 3) (make-array 2 :element-type 'single-float))))
  (check (handler-case (progn (axpy 1 '(1 2) (list 1 2)) nil) (type-error () t))
         "axpy should refuse a Y that is not a single-float vector"))

(section "datasets: shape, labels, determinism, lifetime"
  (with-data ((d (make-spirals :seed 7)) (same (make-spirals :seed 7)) (other (make-spirals :seed 8)))
    (check (= (data-size d) 600))
    (check (= (data-dim d) 2))
    (check (= (data-classes d) 3))
    (check (loop for i below (data-size d) always (< (data-label d i) 3)))
    (check (loop for i below (data-size d) always (every (lambda (v) (< (abs v) 1.1)) (data-point d i))))
    (check (equal (data-point d 5) (data-point same 5)) "same seed must give the same points")
    (check (not (equal (data-point d 5) (data-point other 5))) "different seeds must differ")
    (check (signals-tinyai-error (data-point d 600))))
  (let ((d (make-spirals :per-class 10)))
    (free-data d)
    (free-data d)                                  ; second free must be harmless
    (check (signals-tinyai-error (data-point d 0)) "use after free must be an error, not a crash"))
  (check (handler-case (progn (make-spirals :per-class 0) nil) (type-error () t))))

(section "networks: construction, validation, predict / logits, lifetime"
  (with-net ((n (make-net '(2 64 64 3) :seed 3)))
    (check (= (net-params n) 4547))
    (check (equal (net-dims n) '(2 64 64 3)))
    (check (string= (net-backend n) (kernel-name)))
    (let* ((x '(0.3 -0.4))
           (scores (logits n x))
           (best (position (reduce #'max scores) scores)))
      (check (= (length scores) 3))
      (check (= (predict n x) best) "predict must be the argmax of logits"))
    (check (signals-tinyai-error (predict n '(1 2 3))) "wrong input length"))
  (check (signals-tinyai-error (make-net '(2))))
  (check (signals-tinyai-error (make-net '(2 0 3))))
  (check (signals-tinyai-error (make-net '(2 :wide 3))))
  (check (signals-tinyai-error (make-net '(1 1 1 1 1 1 1 1 1 1))) "more than 8 layers")
  (let ((n (make-net '(2 8 3))))
    (free-net n)
    (free-net n)
    (check (signals-tinyai-error (predict n '(0 0))) "use after free")))

(section "a dataset that does not fit the network is an error, not an out-of-bounds read"
  (with-data ((d (make-spirals)))
    (with-net ((wrong-inputs (make-net '(3 8 3))) (too-few-outputs (make-net '(2 8 2))))
      (check (signals-tinyai-error (evaluate wrong-inputs d)))
      (check (signals-tinyai-error (train-epoch wrong-inputs d)))
      (check (signals-tinyai-error (evaluate too-few-outputs d)))
      (check (signals-tinyai-error (train-epoch too-few-outputs d))))))

(section "learning works and is deterministic"
  (flet ((run (&key force-scalar)
           (with-data ((train (make-spirals :seed 11)) (test (make-spirals :seed 12)))
             (with-net ((net (make-net '(2 32 32 3) :seed 13 :force-scalar force-scalar)))
               (let ((untrained (nth-value 1 (evaluate net test))))
                 (fit net train :epochs 60 :stream nil)
                 (multiple-value-bind (loss acc) (evaluate net test)
                   (list untrained loss acc (net-backend net))))))))
    (let ((first (run)) (second (run)))
      (check (equal first second) "same seeds must give bit-identical results: ~s vs ~s" first second)
      (check (< (first first) 60) "an untrained network should be near chance, got ~f" (first first))
      (check (> (third first) 93) "test accuracy after 60 epochs: ~f" (third first)))))

(section "AVX2 and scalar kernels learn equally well"
  (if (not (avx2-p))
      (format t "(no AVX2 here) ")
      (flet ((accuracy (force-scalar)
               (with-data ((train (make-spirals :seed 21)) (test (make-spirals :seed 22)))
                 (with-net ((net (make-net '(2 32 32 3) :seed 23 :force-scalar force-scalar)))
                   (fit net train :epochs 100 :stream nil)
                   (values (nth-value 1 (evaluate net test)) (net-backend net))))))
        (multiple-value-bind (avx-acc avx-name) (accuracy nil)
          (multiple-value-bind (scalar-acc scalar-name) (accuracy t)
            (check (string/= avx-name scalar-name) "backends should differ: ~a / ~a" avx-name scalar-name)
            (check (> avx-acc 93) "AVX2 test accuracy ~f" avx-acc)
            (check (> scalar-acc 93) "scalar test accuracy ~f" scalar-acc)
            (check (< (abs (- avx-acc scalar-acc)) 4) "AVX2 ~f vs scalar ~f" avx-acc scalar-acc))))))

;;; --- the headline test: Lisp-driven training == the C program ---------------

(defun c-program-path ()
  (let ((exe (probe-file (merge-pathnames (make-pathname :directory '(:relative :up "build")
                                                         :name "tinyai" :type "exe")
                                          (or *load-truename* *default-pathname-defaults*)))))
    (and exe (sb-ext:native-namestring exe))))

(defun run-c-program (&rest arguments)
  (let ((path (c-program-path)))
    (with-output-to-string (out)
      (sb-ext:run-program path arguments :output out :error nil :input nil :wait t))))

(defun numbers-in (string)
  "All decimal numbers appearing in STRING, as double-floats."
  (let ((*read-default-float-format* 'double-float) (result '()) (i 0) (n (length string)))
    (loop while (< i n)
          do (if (digit-char-p (char string i))
                 (let ((j i))
                   (loop while (and (< j n) (or (digit-char-p (char string j)) (char= (char string j) #\.)))
                         do (incf j))
                   (push (let ((*read-eval* nil)) (read-from-string (subseq string i j))) result)
                   (setf i j))
                 (incf i)))
    (mapcar (lambda (x) (float x 1d0)) (nreverse result))))

(defun lines-of (string)
  (with-input-from-string (s string)
    (loop for line = (read-line s nil) while line collect (string-right-trim '(#\Return) line))))

(section "training from Lisp reproduces the C program (accuracy, loss and the picture)"
  (if (not (c-program-path))
      (format t "(build\\tinyai.exe not found - skipped) ")
      (let* ((c-lines (lines-of (run-c-program "--log-every" "300")))
             (c-final (numbers-in (find-if (lambda (l) (eql 0 (search "final" l))) c-lines)))
             (c-grid (remove-if-not (lambda (l) (and (plusp (length l)) (char= (char l 0) #\|))) c-lines))
             (lisp-plot nil)
             (result nil))
        (with-data ((train (make-spirals :per-class 200 :turns 1.0 :noise 0.4 :seed 1234))
                    (test  (make-spirals :per-class 200 :turns 1.0 :noise 0.4 :seed 1235)))
          (with-net ((net (make-net '(2 64 64 3) :seed 1236)))
            (setf result (fit net train :epochs 300 :lr 0.02 :lr-final 0.05 :batch 1 :test test :stream nil))
            (setf lisp-plot (lines-of (with-output-to-string (s) (plot-regions net test :stream s))))))
        (check (= (length c-final) 3) "could not parse the C program's final line: ~s" c-final)
        (check (< (abs (- (getf result :train-accuracy) (first c-final))) 0.0051)
               "train accuracy: Lisp ~f vs C ~f" (getf result :train-accuracy) (first c-final))
        (check (< (abs (- (getf result :test-accuracy) (second c-final))) 0.0051)
               "test accuracy: Lisp ~f vs C ~f" (getf result :test-accuracy) (second c-final))
        (check (< (abs (- (getf result :test-loss) (third c-final))) 0.00051)
               "test loss: Lisp ~f vs C ~f" (getf result :test-loss) (third c-final))
        (let ((lisp-grid (remove-if-not (lambda (l) (and (plusp (length l)) (char= (char l 0) #\|))) lisp-plot)))
          (check (= (length c-grid) 33) "C picture has ~d rows" (length c-grid))
          (check (equal lisp-grid c-grid) "the Lisp and C pictures differ")))))

(section "handles survive heavy churn and garbage collection"
  (dotimes (i 300)                                  ; explicit lifetime
    (with-data ((d (make-spirals :per-class 5 :seed i)))
      (with-net ((n (make-net '(2 4 3) :seed i)))
        (train-epoch n d))))
  (dotimes (i 200)                                  ; forgotten handles: finalizers must free them
    (make-net '(2 8 3) :seed i)
    (make-spirals :per-class 5 :seed i))
  (sb-ext:gc :full t)
  (sleep 0.3)
  (check t))

;;; ---------------------------------------------------------------------------

(format t "~%~d checks, ~d failed -> ~a~%" *checks* *failures* (if (zerop *failures*) "PASS" "FAIL"))
(finish-output)
(sb-ext:exit :code (if (zerop *failures*) 0 1) :abort t)
