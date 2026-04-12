// tensor/view/shapetracker.c — ShapeTracker composition helpers
// Included after view_create/view_reshape/view_permute are defined.

// Compose a reshape onto a ShapeTracker. If merge succeeds, replaces last view.
// If merge fails (view_reshape returns numel=0), pushes a new contiguous view.
static ShapeTracker st_reshape(ShapeTracker st, Shape new_shape) {
    if (st.n_views == 0) return st_from_view(view_create(new_shape));
    // Verify numel match before calling view_reshape (which asserts)
    u32 old_numel = st.views[st.n_views - 1].numel;
    u32 new_numel = 1;
    for (u32 i = 0; i < new_shape.rank; i++) new_numel *= new_shape.dims[i];
    if (old_numel != new_numel) {
        // Numel mismatch — can't reshape. Return unchanged.
        // Caller must handle (e.g., create a boundary in the fuser).
        return st;
    }
    View merged = view_reshape(st.views[st.n_views - 1], new_shape);
    if (merged.numel != 0) {
        st.views[st.n_views - 1] = merged;
    } else {
        // Can't merge — push new contiguous view (like tinygrad)
        if (st.n_views < ST_MAX_VIEWS)
            st.views[st.n_views++] = view_create(new_shape);
    }
    return st;
}

// Compose permute onto last view (always merges).
static ShapeTracker st_permute(ShapeTracker st, const u32 *axes, u32 rank) {
    if (st.n_views == 0) return st;
    st.views[st.n_views - 1] = view_permute(st.views[st.n_views - 1], axes);
    return st;
}

// Compose expand onto last view (always merges — sets stride=0 for broadcast dims).
static ShapeTracker st_expand(ShapeTracker st, const u32 *dims, u32 rank) {
    if (st.n_views == 0) return st;
    View *v = &st.views[st.n_views - 1];
    View nv = *v; nv.shape.rank = rank; nv.numel = 1;
    for (u32 i = 0; i < rank; i++) {
        u32 nd = dims[i];
        if (i < v->shape.rank && v->shape.dims[i] == 1 && nd > 1) nv.strides[i] = 0;
        nv.shape.dims[i] = nd; nv.numel *= nd;
    }
    nv.contiguous = 0;
    st.views[st.n_views - 1] = nv;
    return st;
}

// Compose shrink onto last view (always merges).
static ShapeTracker st_shrink(ShapeTracker st, const u32 *starts, const u32 *ends, u32 ndim) {
    if (st.n_views == 0) return st;
    st.views[st.n_views - 1] = view_shrink(st.views[st.n_views - 1], starts, ends);
    return st;
}

// Compose pad onto last view (always merges).
static ShapeTracker st_pad(ShapeTracker st, const u32 *pad_before, const u32 *pad_after) {
    if (st.n_views == 0) return st;
    st.views[st.n_views - 1] = view_pad(st.views[st.n_views - 1], pad_before, pad_after);
    return st;
}
