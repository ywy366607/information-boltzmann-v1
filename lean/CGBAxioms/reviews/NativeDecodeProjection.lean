import Init

/-! Runtime binding for gdn_decode_step.py:
  G = q,k,v,g,alpha,beta; I = normalized input coordinates.
  W/B are copied from live layer projections into stable graph buffers at
  start(). No parameters, roles, ports, history or recurrent terms are merged.
  The theorem concerns the real arithmetic projection, not FP32 rounding.
  The convolution/delta/admission/MLP map is tested against gdn_fused at the
  actual d256/H8/M768/C4 configuration and across recurrent events.
-/
namespace CGB.NativeDecodeProjection
variable {G O I R : Type}

def projection (add mul : R → R → R) (zero : R) (indices : List I)
    (w : O → I → R) (b : O → R) (x : I → R) (o : O) : R :=
  add (indices.foldl (fun s i => add s (mul (w o i) (x i))) zero) (b o)

theorem packed_projection_preserves_each_role
    (add mul : R → R → R) (zero : R) (indices : List I)
    (w : G → O → I → R) (b : G → O → R) (x : I → R) (g : G) (o : O) :
    projection add mul zero indices (fun p : G × O => w p.1 p.2)
      (fun p => b p.1 p.2) x (g, o) =
      projection add mul zero indices (w g) (b g) x o := rfl

-- Layer-depth composition reuses NativeWordBatch.original_layers_batch_pointwise.
end CGB.NativeDecodeProjection
