use pdf_writer::{Chunk, Finish, Name, Pdf, Ref, Str, TextStr};
use std::collections::HashMap;
use std::sync::OnceLock;
use xmp_writer::{RenditionClass, XmpWriter};

use crate::configure::{PdfVersion, ValidationError};
use crate::error::KrillaResult;
use crate::interchange::metadata::Metadata;
use crate::metadata::PageLayout;
use crate::serialize::SerializeContext;
use crate::util::{stable_hash_base64, Deferred};

type DChunk = Deferred<Chunk>;

pub(crate) type ChunkContainerFn = fn(&mut ChunkContainer) -> &mut Vec<DChunk>;

/// Collects all chunks that we create while building
/// the PDF and then writes them out in an orderly manner.
#[derive(Default)]
pub(crate) struct ChunkContainer {
    pub(crate) page_tree: Option<(Ref, Chunk)>,
    pub(crate) outline: Option<(Ref, Chunk)>,
    pub(crate) page_label_tree: Option<(Ref, Chunk)>,
    pub(crate) destination_profiles: Option<(Ref, Chunk)>,
    pub(crate) struct_tree_root: Option<(Ref, Chunk)>,

    pub(crate) struct_elements: Vec<Chunk>,
    pub(crate) page_labels: Vec<Chunk>,
    pub(crate) annotations: Vec<Chunk>,
    pub(crate) fonts: Vec<Chunk>,
    pub(crate) color_spaces: Vec<DChunk>,
    pub(crate) icc_profiles: Vec<DChunk>,
    pub(crate) destinations: Vec<Chunk>,
    pub(crate) ext_g_states: Vec<DChunk>,
    pub(crate) masks: Vec<DChunk>,
    pub(crate) x_objects: Vec<DChunk>,
    pub(crate) shading_functions: Vec<DChunk>,
    pub(crate) patterns: Vec<DChunk>,
    pub(crate) pages: Vec<DChunk>,
    pub(crate) images: Vec<Deferred<KrillaResult<Chunk>>>,
    pub(crate) embedded_files: Vec<DChunk>,
    pub(crate) embedded_pdfs: Vec<Deferred<KrillaResult<EmbeddedPdfChunk>>>,

    pub(crate) metadata: Option<Metadata>,
}

impl ChunkContainer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn finish(mut self, sc: &mut SerializeContext) -> KrillaResult<Pdf> {
        let mut remapped_ref = Ref::new(1);
        let mut remapper = HashMap::new();

        // Allows us to estimate the capacity we will need for the new PDF.
        let mut chunks_byte_len = 0;

        // This traverses the chunks in the order that we will write them to the PDF and assigns new
        // references as we go. This gives us the advantage that the PDF will be numbered with
        // monotonically increasing numbers, which, while it is not a strict requirement for a valid
        // PDF, makes it a lot cleaner and might make implementing features like object streams
        // easier down the road.
        //
        // It also allows us to estimate the capacity we will need for the new PDF.
        self.visit(sc, &mut |chunk| {
            for object_ref in chunk.refs() {
                let existing = remapper.insert(object_ref, remapped_ref.bump());
                debug_assert!(existing.is_none());
            }
            chunks_byte_len += chunk.len();
        })?;

        // Chunk length is not an exact number because the length might change as we renumber,
        // so we add a bit of a padding by multiplying with 1.1. The 200 is additional padding
        // for the document catalog. This hopefully allows us to avoid re-allocations in the general
        // case, and thus give us better performance.
        let capacity = (chunks_byte_len as f32 * 1.1 + 200.0) as usize;
        let mut pdf = Pdf::with_capacity(capacity);
        sc.serialize_settings().pdf_version().set_version(&mut pdf);

        if sc.serialize_settings().ascii_compatible
            && !sc.serialize_settings().validator().requires_binary_header()
        {
            pdf.set_binary_marker(b"AAAA")
        }

        // Extract catalog-related info BEFORE consuming chunks, since
        // visit_consuming moves self and drops all chunk data.
        let page_tree_ref = self.page_tree.as_ref().map(|(r, _)| remapper[r]);
        let outline_ref = self.outline.as_ref().map(|(r, _)| remapper[r]);
        let page_label_tree_ref = self.page_label_tree.as_ref().map(|(r, _)| remapper[r]);
        let destination_profiles_ref = self.destination_profiles.as_ref().map(|(r, _)| remapper[r]);
        let struct_tree_root_ref = self.struct_tree_root.as_ref().map(|(r, _)| remapper[r]);
        let has_catalog = page_tree_ref.is_some()
            || outline_ref.is_some()
            || page_label_tree_ref.is_some()
            || destination_profiles_ref.is_some()
            || struct_tree_root_ref.is_some();
        let num_pages = self.pages.len() as u32;
        let metadata = self.metadata.take();
        let has_no_embedded_files = self.embedded_files.is_empty();

        // Write chunks into the PDF, consuming (dropping) each field after
        // renumbering to free memory before the PDF buffer grows large.
        // This avoids the struct_elements (268 MB) and final PDF (257 MB)
        // coexisting simultaneously.
        self.visit_consuming(sc, &mut pdf, &remapper)?;

        let missing_title = metadata.as_ref().is_none_or(|m| m.title.is_none());

        if missing_title {
            sc.register_validation_error(ValidationError::NoDocumentTitle);
        }

        // Write the PDF document info metadata.
        if let Some(ref meta) = metadata {
            meta.serialize_document_info(
                &mut remapped_ref,
                &mut pdf,
                sc.serialize_settings().configuration,
            );
        }

        let instance_id = stable_hash_base64(pdf.as_bytes());

        let document_id = if let Some(ref meta) = metadata {
            if let Some(document_id) = &meta.document_id {
                stable_hash_base64(&(sc.serialize_settings().pdf_version().as_str(), document_id))
            } else if meta.title.is_some() && meta.authors.is_some() {
                stable_hash_base64(&(
                    sc.serialize_settings().pdf_version().as_str(),
                    &meta.title,
                    &meta.authors,
                ))
            } else {
                instance_id.clone()
            }
        } else {
            instance_id.clone()
        };

        let mut xmp = XmpWriter::new();
        if let Some(ref meta) = metadata {
            meta.serialize_xmp_metadata(&mut xmp, sc, &instance_id);
        }

        sc.serialize_settings().validator().write_xmp(&mut xmp);

        xmp.num_pages(num_pages);
        xmp.format("application/pdf");
        xmp.instance_id(&instance_id);
        xmp.document_id(&document_id);
        pdf.set_file_id((
            document_id.as_bytes().to_vec(),
            instance_id.as_bytes().to_vec(),
        ));

        xmp.rendition_class(RenditionClass::Proof);
        sc.serialize_settings().pdf_version().write_xmp(&mut xmp);

        let named_destinations = sc.global_objects.named_destinations.take();
        let embedded_files = sc.global_objects.embedded_files.take();

        // We only write a catalog if a page tree exists. Every valid PDF must have one
        // and krilla ensures that there always is one, but for snapshot tests, it can be
        // useful to not write a document catalog if we don’t actually need it for the test.
        if has_catalog {
            let meta_ref = if sc.serialize_settings().xmp_metadata {
                let meta_ref = remapped_ref.bump();
                let xmp_buf = xmp.finish(None);
                pdf.stream(meta_ref, xmp_buf.as_bytes())
                    .pair(Name(b"Type"), Name(b"Metadata"))
                    .pair(Name(b"Subtype"), Name(b"XML"));
                Some(meta_ref)
            } else {
                None
            };

            let catalog_ref = remapped_ref.bump();

            let mut catalog = pdf.catalog(catalog_ref);

            if let Some(pt_ref) = page_tree_ref {
                catalog.pages(pt_ref);
            }

            if let Some(meta_ref) = meta_ref {
                catalog.metadata(meta_ref);
            }

            if let Some(pl_ref) = page_label_tree_ref {
                catalog.pair(Name(b"PageLabels"), pl_ref);
            }

            if let Some(oi_ref) = destination_profiles_ref {
                catalog.pair(Name(b"OutputIntents"), oi_ref);
            }

            if let Some(lang) = metadata.as_ref().and_then(|m| m.language.as_ref()) {
                catalog.lang(TextStr(lang));
            } else {
                sc.register_validation_error(ValidationError::NoDocumentLanguage);
            }

            if let Some(st_ref) = struct_tree_root_ref {
                catalog.pair(Name(b"StructTreeRoot"), st_ref);
                let mut mark_info = catalog.mark_info();
                mark_info.marked(true);
                if sc.serialize_settings().pdf_version() >= PdfVersion::Pdf16
                    && sc.serialize_settings().pdf_version() < PdfVersion::Pdf20
                {
                    // We always set suspects to false because it’s required by PDF/UA.
                    mark_info.suspects(false);
                }
                mark_info.finish();
            }

            let write_doc_title = sc
                .serialize_settings()
                .validator()
                .requires_display_doc_title();
            let text_direction = metadata.as_ref().and_then(|m| m.text_direction);

            if write_doc_title || text_direction.is_some() {
                let mut vp = catalog.viewer_preferences();

                if write_doc_title {
                    vp.display_doc_title(true);
                }

                if let Some(dir) = text_direction {
                    vp.direction(dir.to_pdf());
                }
            }

            let page_layout = metadata.as_ref().and_then(|m| m.page_layout);
            if let Some(layout) = page_layout {
                // TwoPageLeft and TwoPageRight are only available PDF 1.5+
                if sc.serialize_settings().pdf_version() >= PdfVersion::Pdf15
                    || !matches!(layout, PageLayout::TwoPageLeft | PageLayout::TwoPageRight)
                {
                    catalog.page_layout(layout.to_pdf());
                }
            }

            if let Some(ol_ref) = outline_ref {
                catalog.outlines(ol_ref);
            }

            let write_embedded_files = sc
                .serialize_settings()
                .validator()
                .write_embedded_files(has_no_embedded_files);

            if !named_destinations.is_empty() || write_embedded_files {
                // Cannot use pdf-writer API here because it requires Ref’s, while
                // we write our destinations directly into the array.
                let mut names = catalog.names();

                if !named_destinations.is_empty() {
                    let mut dest_name_tree = names.destinations();
                    let mut dest_name_entries = dest_name_tree.names();

                    // "The Names entries in the leaf (or root) nodes shall
                    // contain the tree’s keys and their associated values,
                    // arranged in key-value pairs and shall be sorted lexically
                    // in ascending order by key. Shorter keys shall appear
                    // before longer ones beginning with the same byte sequence.
                    // Any encoding of the keys may be used as long as it is
                    // self-consistent; keys shall be compared for equality on
                    // a simple byte-by-byte basis."
                    let mut sorted = named_destinations.into_iter().collect::<Vec<_>>();
                    sorted.sort_by(|a, b| a.0.name.as_bytes().cmp(b.0.name.as_bytes()));

                    for (name, dest_ref) in sorted {
                        dest_name_entries.insert(Str(name.name.as_bytes()), remapper[&dest_ref]);
                    }

                    dest_name_entries.finish();
                    dest_name_tree.finish();
                }

                if write_embedded_files {
                    let mut embedded_files_name_tree = names.embedded_files();
                    let mut embedded_name_entries = embedded_files_name_tree.names();

                    for (name, _ref) in &embedded_files {
                        embedded_name_entries.insert(Str(name.as_bytes()), remapper[_ref]);
                    }
                }
            }

            if !embedded_files.is_empty()
                && sc
                    .serialize_settings()
                    .validator()
                    .allows_associated_files()
            {
                let mut associated_files = catalog.insert(Name(b"AF")).array().typed();
                for _ref in embedded_files.values() {
                    associated_files.item(remapper[_ref]).finish();
                }
            }

            catalog.finish();
        }

        Ok(pdf)
    }
}

pub(crate) struct EmbeddedPdfChunk {
    pub(crate) original_chunk: Chunk,
    pub(crate) root_ref_mappings: HashMap<Ref, Ref>,
    pub(crate) new_chunk: OnceLock<Chunk>,
}

/// Visits all chunks in a type.
trait Visit {
    fn visit(&self, sc: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()>;
}

/// Consuming visit: renumbers chunks into the PDF and drops them.
trait VisitConsuming {
    fn visit_consuming(self, sc: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()>;
}

impl VisitConsuming for EmbeddedPdfChunk {
    fn visit_consuming(self, sc: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()> {
        // Use the cached renumbered chunk if available, otherwise renumber now.
        let renumbered = self.new_chunk.into_inner().unwrap_or_else(|| {
            let mut inner_remapper = self.root_ref_mappings;
            self.original_chunk
                .renumber(|old| *inner_remapper.entry(old).or_insert_with(|| sc.new_ref()))
        });
        renumbered.visit_consuming(sc, pdf, remapper)
    }
}

impl Visit for EmbeddedPdfChunk {
    fn visit(&self, sc: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()> {
        // Now, we have a chunk that contains everything we need to fully embed the PDF, including
        // the pages we wanted to extract into, as well as all their dependencies. The
        // problem is: during the document creation, we already assigned references to the
        // pages (stored in `SerializerContex::page_infos`), but `hayro_write` created new references
        // for those (stored in `result.root_refs`).

        // Because of this, embedded PDF chunks will be renumbered twice: First, we preprocess the
        // chunk such that page/XObjects are reassigned their original references from the serialize
        // context, and all other objects are assigned new, unique references provided by the
        // serialize context. Then, we renumber them once again by treating them like any other chunk.

        // Since we are calling `visit` twice, we also cache the renumbered chunk.

        let renumbered = self.new_chunk.get_or_init(|| {
            let mut remapper = self.root_ref_mappings.clone();

            self.original_chunk
                .renumber(|old| *remapper.entry(old).or_insert_with(|| sc.new_ref()))
        });

        renumbered.visit(sc, f)
    }
}

impl Visit for ChunkContainer {
    fn visit(&self, sc: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()> {
        self.page_tree.visit(sc, f)?;
        self.outline.visit(sc, f)?;
        self.page_label_tree.visit(sc, f)?;
        self.destination_profiles.visit(sc, f)?;
        self.struct_tree_root.visit(sc, f)?;
        self.struct_elements.visit(sc, f)?;
        self.page_labels.visit(sc, f)?;
        self.annotations.visit(sc, f)?;
        self.fonts.visit(sc, f)?;
        self.color_spaces.visit(sc, f)?;
        self.icc_profiles.visit(sc, f)?;
        self.destinations.visit(sc, f)?;
        self.ext_g_states.visit(sc, f)?;
        self.masks.visit(sc, f)?;
        self.x_objects.visit(sc, f)?;
        self.shading_functions.visit(sc, f)?;
        self.patterns.visit(sc, f)?;
        self.pages.visit(sc, f)?;
        self.images.visit(sc, f)?;
        self.embedded_files.visit(sc, f)?;
        self.embedded_pdfs.visit(sc, f)?;
        Ok(())
    }
}

impl ChunkContainer {
    /// Consuming visit: renumber each chunk into the PDF and drop it immediately.
    /// This reduces peak memory by freeing large chunks (struct_elements, pages)
    /// before the PDF buffer grows to its full size.
    fn visit_consuming(self, sc: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()> {
        self.page_tree.visit_consuming(sc, pdf, remapper)?;
        self.outline.visit_consuming(sc, pdf, remapper)?;
        self.page_label_tree.visit_consuming(sc, pdf, remapper)?;
        self.destination_profiles.visit_consuming(sc, pdf, remapper)?;
        self.struct_tree_root.visit_consuming(sc, pdf, remapper)?;
        self.struct_elements.visit_consuming(sc, pdf, remapper)?;
        self.page_labels.visit_consuming(sc, pdf, remapper)?;
        self.annotations.visit_consuming(sc, pdf, remapper)?;
        self.fonts.visit_consuming(sc, pdf, remapper)?;
        self.color_spaces.visit_consuming(sc, pdf, remapper)?;
        self.icc_profiles.visit_consuming(sc, pdf, remapper)?;
        self.destinations.visit_consuming(sc, pdf, remapper)?;
        self.ext_g_states.visit_consuming(sc, pdf, remapper)?;
        self.masks.visit_consuming(sc, pdf, remapper)?;
        self.x_objects.visit_consuming(sc, pdf, remapper)?;
        self.shading_functions.visit_consuming(sc, pdf, remapper)?;
        self.patterns.visit_consuming(sc, pdf, remapper)?;
        self.pages.visit_consuming(sc, pdf, remapper)?;
        self.images.visit_consuming(sc, pdf, remapper)?;
        self.embedded_files.visit_consuming(sc, pdf, remapper)?;
        self.embedded_pdfs.visit_consuming(sc, pdf, remapper)?;
        Ok(())
    }
}

impl Visit for Chunk {
    fn visit(&self, _: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()> {
        f(self);
        Ok(())
    }
}

impl Visit for Option<(Ref, Chunk)> {
    fn visit(&self, sc: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()> {
        if let Some((_, chunk)) = self {
            chunk.visit(sc, f)?;
        }
        Ok(())
    }
}

impl<T: Visit + Send + Sync + 'static> Visit for Deferred<T> {
    fn visit(&self, sc: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()> {
        self.wait().visit(sc, f)
    }
}

impl<T: Visit> Visit for KrillaResult<T> {
    fn visit(&self, sc: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()> {
        self.as_ref().map_err(|e| e.clone())?.visit(sc, f)
    }
}

impl<T: Visit> Visit for Vec<T> {
    fn visit(&self, sc: &mut SerializeContext, f: &mut impl FnMut(&Chunk)) -> KrillaResult<()> {
        for field in self {
            field.visit(sc, f)?;
        }
        Ok(())
    }
}

// --- Consuming visit implementations ---

impl VisitConsuming for Chunk {
    fn visit_consuming(self, _: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()> {
        self.renumber_into(pdf, |old| remapper[&old]);
        Ok(())
    }
}

impl VisitConsuming for Option<(Ref, Chunk)> {
    fn visit_consuming(self, sc: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()> {
        if let Some((_, chunk)) = self {
            chunk.visit_consuming(sc, pdf, remapper)?;
        }
        Ok(())
    }
}

impl<T: Visit + Send + Sync + 'static> VisitConsuming for Deferred<T> {
    fn visit_consuming(self, sc: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()> {
        // Deferred values can't be easily moved out (Arc<OnceCell<T>>).
        // Wait for the value, then use the borrowing Visit to renumber.
        // The Deferred is dropped after this call, freeing the Arc.
        self.wait().visit(sc, &mut |chunk| {
            chunk.renumber_into(pdf, |old| remapper[&old]);
        })
    }
}

impl<T: VisitConsuming> VisitConsuming for KrillaResult<T> {
    fn visit_consuming(self, sc: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()> {
        self?.visit_consuming(sc, pdf, remapper)
    }
}

impl<T: VisitConsuming> VisitConsuming for Vec<T> {
    fn visit_consuming(self, sc: &mut SerializeContext, pdf: &mut Pdf, remapper: &HashMap<Ref, Ref>) -> KrillaResult<()> {
        for field in self {
            field.visit_consuming(sc, pdf, remapper)?;
        }
        Ok(())
    }
}
