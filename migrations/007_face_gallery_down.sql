-- Down migration 007: Remove face gallery
DROP INDEX IF EXISTS idx_face_gallery_model;
DROP INDEX IF EXISTS idx_face_gallery_subject;
DROP TABLE IF EXISTS face_gallery;
