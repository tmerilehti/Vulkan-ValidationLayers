/* Copyright (c) 2015-2019 The Khronos Group Inc.
 * Copyright (c) 2015-2019 Valve Corporation
 * Copyright (c) 2015-2019 LunarG, Inc.
 * Copyright (C) 2015-2019 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Tobin Ehlis <tobine@google.com>
 *         John Zulauf <jzulauf@lunarg.com>
 */

// Allow use of STL min and max functions in Windows
#define NOMINMAX

#include "chassis.h"
#include "core_validation_error_enums.h"
#include "core_validation.h"
#include "descriptor_sets.h"
#include "hash_vk_types.h"
#include "vk_enum_string_helper.h"
#include "vk_safe_struct.h"
#include "vk_typemap_helper.h"
#include "buffer_validation.h"
#include <sstream>
#include <algorithm>
#include <array>
#include <memory>

// ExtendedBinding collects a VkDescriptorSetLayoutBinding and any extended
// state that comes from a different array/structure so they can stay together
// while being sorted by binding number.
struct ExtendedBinding {
    ExtendedBinding(const VkDescriptorSetLayoutBinding *l, VkDescriptorBindingFlagsEXT f) : layout_binding(l), binding_flags(f) {}

    const VkDescriptorSetLayoutBinding *layout_binding;
    VkDescriptorBindingFlagsEXT binding_flags;
};

struct BindingNumCmp {
    bool operator()(const ExtendedBinding &a, const ExtendedBinding &b) const {
        return a.layout_binding->binding < b.layout_binding->binding;
    }
};

using DescriptorSetLayoutDef = cvdescriptorset::DescriptorSetLayoutDef;
using DescriptorSetLayoutId = cvdescriptorset::DescriptorSetLayoutId;

// Canonical dictionary of DescriptorSetLayoutDef (without any handle/device specific information)
cvdescriptorset::DescriptorSetLayoutDict descriptor_set_layout_dict;

DescriptorSetLayoutId GetCanonicalId(const VkDescriptorSetLayoutCreateInfo *p_create_info) {
    return descriptor_set_layout_dict.look_up(DescriptorSetLayoutDef(p_create_info));
}

// Construct DescriptorSetLayout instance from given create info
// Proactively reserve and resize as possible, as the reallocation was visible in profiling
cvdescriptorset::DescriptorSetLayoutDef::DescriptorSetLayoutDef(const VkDescriptorSetLayoutCreateInfo *p_create_info)
    : flags_(p_create_info->flags), binding_count_(0), descriptor_count_(0), dynamic_descriptor_count_(0) {
    const auto *flags_create_info = lvl_find_in_chain<VkDescriptorSetLayoutBindingFlagsCreateInfoEXT>(p_create_info->pNext);

    binding_type_stats_ = {0, 0, 0};
    std::set<ExtendedBinding, BindingNumCmp> sorted_bindings;
    const uint32_t input_bindings_count = p_create_info->bindingCount;
    // Sort the input bindings in binding number order, eliminating duplicates
    for (uint32_t i = 0; i < input_bindings_count; i++) {
        VkDescriptorBindingFlagsEXT flags = 0;
        if (flags_create_info && flags_create_info->bindingCount == p_create_info->bindingCount) {
            flags = flags_create_info->pBindingFlags[i];
        }
        sorted_bindings.insert(ExtendedBinding(p_create_info->pBindings + i, flags));
    }

    // Store the create info in the sorted order from above
    std::map<uint32_t, uint32_t> binding_to_dyn_count;
    uint32_t index = 0;
    binding_count_ = static_cast<uint32_t>(sorted_bindings.size());
    bindings_.reserve(binding_count_);
    binding_flags_.reserve(binding_count_);
    binding_to_index_map_.reserve(binding_count_);
    for (auto input_binding : sorted_bindings) {
        // Add to binding and map, s.t. it is robust to invalid duplication of binding_num
        const auto binding_num = input_binding.layout_binding->binding;
        binding_to_index_map_[binding_num] = index++;
        bindings_.emplace_back(input_binding.layout_binding);
        auto &binding_info = bindings_.back();
        binding_flags_.emplace_back(input_binding.binding_flags);

        descriptor_count_ += binding_info.descriptorCount;
        if (binding_info.descriptorCount > 0) {
            non_empty_bindings_.insert(binding_num);
        }

        if (binding_info.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
            binding_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
            binding_to_dyn_count[binding_num] = binding_info.descriptorCount;
            dynamic_descriptor_count_ += binding_info.descriptorCount;
            binding_type_stats_.dynamic_buffer_count++;
        } else if ((binding_info.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) ||
                   (binding_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)) {
            binding_type_stats_.non_dynamic_buffer_count++;
        } else {
            binding_type_stats_.image_sampler_count++;
        }
    }
    assert(bindings_.size() == binding_count_);
    assert(binding_flags_.size() == binding_count_);
    uint32_t global_index = 0;
    binding_to_global_index_range_map_.reserve(binding_count_);
    // Vector order is finalized so create maps of bindings to descriptors and descriptors to indices
    for (uint32_t i = 0; i < binding_count_; ++i) {
        auto binding_num = bindings_[i].binding;
        auto final_index = global_index + bindings_[i].descriptorCount;
        binding_to_global_index_range_map_[binding_num] = IndexRange(global_index, final_index);
        if (final_index != global_index) {
            global_start_to_index_map_[global_index] = i;
        }
        global_index = final_index;
    }

    // Now create dyn offset array mapping for any dynamic descriptors
    uint32_t dyn_array_idx = 0;
    binding_to_dynamic_array_idx_map_.reserve(binding_to_dyn_count.size());
    for (const auto &bc_pair : binding_to_dyn_count) {
        binding_to_dynamic_array_idx_map_[bc_pair.first] = dyn_array_idx;
        dyn_array_idx += bc_pair.second;
    }
}

size_t cvdescriptorset::DescriptorSetLayoutDef::hash() const {
    hash_util::HashCombiner hc;
    hc << flags_;
    hc.Combine(bindings_);
    hc.Combine(binding_flags_);
    return hc.Value();
}
//

// Return valid index or "end" i.e. binding_count_;
// The asserts in "Get" are reduced to the set where no valid answer(like null or 0) could be given
// Common code for all binding lookups.
uint32_t cvdescriptorset::DescriptorSetLayoutDef::GetIndexFromBinding(uint32_t binding) const {
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.cend()) return bi_itr->second;
    return GetBindingCount();
}
VkDescriptorSetLayoutBinding const *cvdescriptorset::DescriptorSetLayoutDef::GetDescriptorSetLayoutBindingPtrFromIndex(
    const uint32_t index) const {
    if (index >= bindings_.size()) return nullptr;
    return bindings_[index].ptr();
}
// Return descriptorCount for given index, 0 if index is unavailable
uint32_t cvdescriptorset::DescriptorSetLayoutDef::GetDescriptorCountFromIndex(const uint32_t index) const {
    if (index >= bindings_.size()) return 0;
    return bindings_[index].descriptorCount;
}
// For the given index, return descriptorType
VkDescriptorType cvdescriptorset::DescriptorSetLayoutDef::GetTypeFromIndex(const uint32_t index) const {
    assert(index < bindings_.size());
    if (index < bindings_.size()) return bindings_[index].descriptorType;
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}
// For the given index, return stageFlags
VkShaderStageFlags cvdescriptorset::DescriptorSetLayoutDef::GetStageFlagsFromIndex(const uint32_t index) const {
    assert(index < bindings_.size());
    if (index < bindings_.size()) return bindings_[index].stageFlags;
    return VkShaderStageFlags(0);
}
// Return binding flags for given index, 0 if index is unavailable
VkDescriptorBindingFlagsEXT cvdescriptorset::DescriptorSetLayoutDef::GetDescriptorBindingFlagsFromIndex(
    const uint32_t index) const {
    if (index >= binding_flags_.size()) return 0;
    return binding_flags_[index];
}

// For the given global index, return index
uint32_t cvdescriptorset::DescriptorSetLayoutDef::GetIndexFromGlobalIndex(const uint32_t global_index) const {
    auto start_it = global_start_to_index_map_.upper_bound(global_index);
    uint32_t index = binding_count_;
    assert(start_it != global_start_to_index_map_.cbegin());
    if (start_it != global_start_to_index_map_.cbegin()) {
        --start_it;
        index = start_it->second;
#ifndef NDEBUG
        const auto &range = GetGlobalIndexRangeFromBinding(bindings_[index].binding);
        assert(range.start <= global_index && global_index < range.end);
#endif
    }
    return index;
}

// For the given binding, return the global index range
// As start and end are often needed in pairs, get both with a single hash lookup.
const cvdescriptorset::IndexRange &cvdescriptorset::DescriptorSetLayoutDef::GetGlobalIndexRangeFromBinding(
    const uint32_t binding) const {
    assert(binding_to_global_index_range_map_.count(binding));
    // In error case max uint32_t so index is out of bounds to break ASAP
    const static IndexRange kInvalidRange = {0xFFFFFFFF, 0xFFFFFFFF};
    const auto &range_it = binding_to_global_index_range_map_.find(binding);
    if (range_it != binding_to_global_index_range_map_.end()) {
        return range_it->second;
    }
    return kInvalidRange;
}

// For given binding, return ptr to ImmutableSampler array
VkSampler const *cvdescriptorset::DescriptorSetLayoutDef::GetImmutableSamplerPtrFromBinding(const uint32_t binding) const {
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        return bindings_[bi_itr->second].pImmutableSamplers;
    }
    return nullptr;
}
// Move to next valid binding having a non-zero binding count
uint32_t cvdescriptorset::DescriptorSetLayoutDef::GetNextValidBinding(const uint32_t binding) const {
    auto it = non_empty_bindings_.upper_bound(binding);
    assert(it != non_empty_bindings_.cend());
    if (it != non_empty_bindings_.cend()) return *it;
    return GetMaxBinding() + 1;
}
// For given index, return ptr to ImmutableSampler array
VkSampler const *cvdescriptorset::DescriptorSetLayoutDef::GetImmutableSamplerPtrFromIndex(const uint32_t index) const {
    if (index < bindings_.size()) {
        return bindings_[index].pImmutableSamplers;
    }
    return nullptr;
}
// If our layout is compatible with rh_ds_layout, return true,
//  else return false and fill in error_msg will description of what causes incompatibility
bool cvdescriptorset::DescriptorSetLayout::IsCompatible(DescriptorSetLayout const *const rh_ds_layout,
                                                        std::string *error_msg) const {
    // Trivial case
    if (layout_ == rh_ds_layout->GetDescriptorSetLayout()) return true;
    if (GetLayoutDef() == rh_ds_layout->GetLayoutDef()) return true;
    bool detailed_compat_check =
        GetLayoutDef()->IsCompatible(layout_, rh_ds_layout->GetDescriptorSetLayout(), rh_ds_layout->GetLayoutDef(), error_msg);
    // The detailed check should never tell us mismatching DSL are compatible
    assert(!detailed_compat_check);
    return detailed_compat_check;
}

// Do a detailed compatibility check of this def (referenced by ds_layout), vs. the rhs (layout and def)
// Should only be called if trivial accept has failed, and in that context should return false.
bool cvdescriptorset::DescriptorSetLayoutDef::IsCompatible(VkDescriptorSetLayout ds_layout, VkDescriptorSetLayout rh_ds_layout,
                                                           DescriptorSetLayoutDef const *const rh_ds_layout_def,
                                                           std::string *error_msg) const {
    if (descriptor_count_ != rh_ds_layout_def->descriptor_count_) {
        std::stringstream error_str;
        error_str << "DescriptorSetLayout " << ds_layout << " has " << descriptor_count_ << " descriptors, but DescriptorSetLayout "
                  << rh_ds_layout << ", which comes from pipelineLayout, has " << rh_ds_layout_def->descriptor_count_
                  << " descriptors.";
        *error_msg = error_str.str();
        return false;  // trivial fail case
    }

    // Descriptor counts match so need to go through bindings one-by-one
    //  and verify that type and stageFlags match
    for (auto binding : bindings_) {
        // TODO : Do we also need to check immutable samplers?
        // VkDescriptorSetLayoutBinding *rh_binding;
        if (binding.descriptorCount != rh_ds_layout_def->GetDescriptorCountFromBinding(binding.binding)) {
            std::stringstream error_str;
            error_str << "Binding " << binding.binding << " for DescriptorSetLayout " << ds_layout << " has a descriptorCount of "
                      << binding.descriptorCount << " but binding " << binding.binding << " for DescriptorSetLayout "
                      << rh_ds_layout << ", which comes from pipelineLayout, has a descriptorCount of "
                      << rh_ds_layout_def->GetDescriptorCountFromBinding(binding.binding);
            *error_msg = error_str.str();
            return false;
        } else if (binding.descriptorType != rh_ds_layout_def->GetTypeFromBinding(binding.binding)) {
            std::stringstream error_str;
            error_str << "Binding " << binding.binding << " for DescriptorSetLayout " << ds_layout << " is type '"
                      << string_VkDescriptorType(binding.descriptorType) << "' but binding " << binding.binding
                      << " for DescriptorSetLayout " << rh_ds_layout << ", which comes from pipelineLayout, is type '"
                      << string_VkDescriptorType(rh_ds_layout_def->GetTypeFromBinding(binding.binding)) << "'";
            *error_msg = error_str.str();
            return false;
        } else if (binding.stageFlags != rh_ds_layout_def->GetStageFlagsFromBinding(binding.binding)) {
            std::stringstream error_str;
            error_str << "Binding " << binding.binding << " for DescriptorSetLayout " << ds_layout << " has stageFlags "
                      << binding.stageFlags << " but binding " << binding.binding << " for DescriptorSetLayout " << rh_ds_layout
                      << ", which comes from pipelineLayout, has stageFlags "
                      << rh_ds_layout_def->GetStageFlagsFromBinding(binding.binding);
            *error_msg = error_str.str();
            return false;
        }
    }
    return true;
}

bool cvdescriptorset::DescriptorSetLayoutDef::IsNextBindingConsistent(const uint32_t binding) const {
    if (!binding_to_index_map_.count(binding + 1)) return false;
    auto const &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        const auto &next_bi_itr = binding_to_index_map_.find(binding + 1);
        if (next_bi_itr != binding_to_index_map_.end()) {
            auto type = bindings_[bi_itr->second].descriptorType;
            auto stage_flags = bindings_[bi_itr->second].stageFlags;
            auto immut_samp = bindings_[bi_itr->second].pImmutableSamplers ? true : false;
            auto flags = binding_flags_[bi_itr->second];
            if ((type != bindings_[next_bi_itr->second].descriptorType) ||
                (stage_flags != bindings_[next_bi_itr->second].stageFlags) ||
                (immut_samp != (bindings_[next_bi_itr->second].pImmutableSamplers ? true : false)) ||
                (flags != binding_flags_[next_bi_itr->second])) {
                return false;
            }
            return true;
        }
    }
    return false;
}
// Starting at offset descriptor of given binding, parse over update_count
//  descriptor updates and verify that for any binding boundaries that are crossed, the next binding(s) are all consistent
//  Consistency means that their type, stage flags, and whether or not they use immutable samplers matches
//  If so, return true. If not, fill in error_msg and return false
bool cvdescriptorset::DescriptorSetLayoutDef::VerifyUpdateConsistency(uint32_t current_binding, uint32_t offset,
                                                                      uint32_t update_count, const char *type,
                                                                      const VkDescriptorSet set, std::string *error_msg) const {
    // Verify consecutive bindings match (if needed)
    auto orig_binding = current_binding;
    // Track count of descriptors in the current_bindings that are remaining to be updated
    auto binding_remaining = GetDescriptorCountFromBinding(current_binding);
    // First, it's legal to offset beyond your own binding so handle that case
    //  Really this is just searching for the binding in which the update begins and adjusting offset accordingly
    while (offset >= binding_remaining) {
        // Advance to next binding, decrement offset by binding size
        offset -= binding_remaining;
        binding_remaining = GetDescriptorCountFromBinding(++current_binding);
    }
    binding_remaining -= offset;
    while (update_count > binding_remaining) {  // While our updates overstep current binding
        // Verify next consecutive binding matches type, stage flags & immutable sampler use
        if (!IsNextBindingConsistent(current_binding++)) {
            std::stringstream error_str;
            error_str << "Attempting " << type;
            if (IsPushDescriptor()) {
                error_str << " push descriptors";
            } else {
                error_str << " descriptor set " << set;
            }
            error_str << " binding #" << orig_binding << " with #" << update_count
                      << " descriptors being updated but this update oversteps the bounds of this binding and the next binding is "
                         "not consistent with current binding so this update is invalid.";
            *error_msg = error_str.str();
            return false;
        }
        // For sake of this check consider the bindings updated and grab count for next binding
        update_count -= binding_remaining;
        binding_remaining = GetDescriptorCountFromBinding(current_binding);
    }
    return true;
}

// The DescriptorSetLayout stores the per handle data for a descriptor set layout, and references the common defintion for the
// handle invariant portion
cvdescriptorset::DescriptorSetLayout::DescriptorSetLayout(const VkDescriptorSetLayoutCreateInfo *p_create_info,
                                                          const VkDescriptorSetLayout layout)
    : layout_(layout), layout_destroyed_(false), layout_id_(GetCanonicalId(p_create_info)) {}


cvdescriptorset::SamplerDescriptor::SamplerDescriptor(const VkSampler *immut) : sampler_(VK_NULL_HANDLE), immutable_(false) {
    updated = false;
    descriptor_class = PlainSampler;
    if (immut) {
        sampler_ = *immut;
        immutable_ = true;
        updated = true;
    }
}


cvdescriptorset::AllocateDescriptorSetsData::AllocateDescriptorSetsData(uint32_t count)
    : required_descriptors_by_type{}, layout_nodes(count, nullptr) {}

cvdescriptorset::DescriptorSet::DescriptorSet(const VkDescriptorSet set, const VkDescriptorPool pool,
                                              const std::shared_ptr<DescriptorSetLayout const> &layout, uint32_t variable_count,
                                              CoreChecks *dev_data)
    : some_update_(false),
      set_(set),
      pool_state_(nullptr),
      p_layout_(layout),
      device_data_(dev_data),
      limits_(dev_data->phys_dev_props.limits),
      variable_count_(variable_count) {
    pool_state_ = dev_data->GetDescriptorPoolState(pool);
    // Foreach binding, create default descriptors of given type
    descriptors_.reserve(p_layout_->GetTotalDescriptorCount());
    for (uint32_t i = 0; i < p_layout_->GetBindingCount(); ++i) {
        auto type = p_layout_->GetTypeFromIndex(i);
        switch (type) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: {
                auto immut_sampler = p_layout_->GetImmutableSamplerPtrFromIndex(i);
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di) {
                    if (immut_sampler) {
                        descriptors_.emplace_back(new SamplerDescriptor(immut_sampler + di));
                        some_update_ = true;  // Immutable samplers are updated at creation
                    } else
                        descriptors_.emplace_back(new SamplerDescriptor(nullptr));
                }
                break;
            }
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                auto immut = p_layout_->GetImmutableSamplerPtrFromIndex(i);
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di) {
                    if (immut) {
                        descriptors_.emplace_back(new ImageSamplerDescriptor(immut + di));
                        some_update_ = true;  // Immutable samplers are updated at creation
                    } else
                        descriptors_.emplace_back(new ImageSamplerDescriptor(nullptr));
                }
                break;
            }
            // ImageDescriptors
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new ImageDescriptor(type));
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new TexelDescriptor(type));
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new BufferDescriptor(type));
                break;
            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new InlineUniformDescriptor(type));
                break;
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new AccelerationStructureDescriptor(type));
                break;
            default:
                assert(0);  // Bad descriptor type specified
                break;
        }
    }
}

cvdescriptorset::DescriptorSet::~DescriptorSet() { InvalidateBoundCmdBuffers(); }

static std::string StringDescriptorReqViewType(descriptor_req req) {
    std::string result("");
    for (unsigned i = 0; i <= VK_IMAGE_VIEW_TYPE_END_RANGE; i++) {
        if (req & (1 << i)) {
            if (result.size()) result += ", ";
            result += string_VkImageViewType(VkImageViewType(i));
        }
    }

    if (!result.size()) result = "(none)";

    return result;
}

static char const *StringDescriptorReqComponentType(descriptor_req req) {
    if (req & DESCRIPTOR_REQ_COMPONENT_TYPE_SINT) return "SINT";
    if (req & DESCRIPTOR_REQ_COMPONENT_TYPE_UINT) return "UINT";
    if (req & DESCRIPTOR_REQ_COMPONENT_TYPE_FLOAT) return "FLOAT";
    return "(none)";
}

// Is this sets underlying layout compatible with passed in layout according to "Pipeline Layout Compatibility" in spec?
bool cvdescriptorset::DescriptorSet::IsCompatible(DescriptorSetLayout const *const layout, std::string *error) const {
    return layout->IsCompatible(p_layout_.get(), error);
}

static unsigned DescriptorRequirementsBitsFromFormat(VkFormat fmt) {
    if (FormatIsSInt(fmt)) return DESCRIPTOR_REQ_COMPONENT_TYPE_SINT;
    if (FormatIsUInt(fmt)) return DESCRIPTOR_REQ_COMPONENT_TYPE_UINT;
    if (FormatIsDepthAndStencil(fmt)) return DESCRIPTOR_REQ_COMPONENT_TYPE_FLOAT | DESCRIPTOR_REQ_COMPONENT_TYPE_UINT;
    if (fmt == VK_FORMAT_UNDEFINED) return 0;
    // everything else -- UNORM/SNORM/FLOAT/USCALED/SSCALED is all float in the shader.
    return DESCRIPTOR_REQ_COMPONENT_TYPE_FLOAT;
}


// Set is being deleted or updates so invalidate all bound cmd buffers
void cvdescriptorset::DescriptorSet::InvalidateBoundCmdBuffers() {
    device_data_->InvalidateCommandBuffers(cb_bindings, {HandleToUint64(set_), kVulkanObjectTypeDescriptorSet});
}

// Loop through the write updates to do for a push descriptor set, ignoring dstSet
void cvdescriptorset::DescriptorSet::PerformPushDescriptorsUpdate(uint32_t write_count, const VkWriteDescriptorSet *p_wds) {
    assert(IsPushDescriptor());
    for (uint32_t i = 0; i < write_count; i++) {
        PerformWriteUpdate(&p_wds[i]);
    }
}

// Perform write update in given update struct
void cvdescriptorset::DescriptorSet::PerformWriteUpdate(const VkWriteDescriptorSet *update) {
    // Perform update on a per-binding basis as consecutive updates roll over to next binding
    auto descriptors_remaining = update->descriptorCount;
    auto binding_being_updated = update->dstBinding;
    auto offset = update->dstArrayElement;
    uint32_t update_index = 0;
    while (descriptors_remaining) {
        uint32_t update_count = std::min(descriptors_remaining, GetDescriptorCountFromBinding(binding_being_updated));
        auto global_idx = p_layout_->GetGlobalIndexRangeFromBinding(binding_being_updated).start + offset;
        // Loop over the updates for a single binding at a time
        for (uint32_t di = 0; di < update_count; ++di, ++update_index) {
            descriptors_[global_idx + di]->WriteUpdate(update, update_index);
        }
        // Roll over to next binding in case of consecutive update
        descriptors_remaining -= update_count;
        offset = 0;
        binding_being_updated++;
    }
    if (update->descriptorCount) some_update_ = true;

    if (!(p_layout_->GetDescriptorBindingFlagsFromBinding(update->dstBinding) &
          (VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT))) {
        InvalidateBoundCmdBuffers();
    }
}
// Perform Copy update
void cvdescriptorset::DescriptorSet::PerformCopyUpdate(const VkCopyDescriptorSet *update, const DescriptorSet *src_set) {
    auto src_start_idx = src_set->GetGlobalIndexRangeFromBinding(update->srcBinding).start + update->srcArrayElement;
    auto dst_start_idx = p_layout_->GetGlobalIndexRangeFromBinding(update->dstBinding).start + update->dstArrayElement;
    // Update parameters all look good so perform update
    for (uint32_t di = 0; di < update->descriptorCount; ++di) {
        auto src = src_set->descriptors_[src_start_idx + di].get();
        auto dst = descriptors_[dst_start_idx + di].get();
        if (src->updated) {
            dst->CopyUpdate(src);
            some_update_ = true;
        } else {
            dst->updated = false;
        }
    }

    if (!(p_layout_->GetDescriptorBindingFlagsFromBinding(update->dstBinding) &
          (VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT))) {
        InvalidateBoundCmdBuffers();
    }
}

// Update the drawing state for the affected descriptors.
// Set cb_node to this set and this set to cb_node.
// Add the bindings of the descriptor
// Set the layout based on the current descriptor layout (will mask subsequent layer mismatch errors)
// TODO: Modify the UpdateDrawState virtural functions to *only* set initial layout and not change layouts
// Prereq: This should be called for a set that has been confirmed to be active for the given cb_node, meaning it's going
//   to be used in a draw by the given cb_node
void cvdescriptorset::DescriptorSet::UpdateDrawState(CoreChecks *device_data, CMD_BUFFER_STATE *cb_node,
                                                     const std::map<uint32_t, descriptor_req> &binding_req_map) {
    // bind cb to this descriptor set
    cb_bindings.insert(cb_node);
    // Add bindings for descriptor set, the set's pool, and individual objects in the set
    cb_node->object_bindings.insert({HandleToUint64(set_), kVulkanObjectTypeDescriptorSet});
    pool_state_->cb_bindings.insert(cb_node);
    cb_node->object_bindings.insert({HandleToUint64(pool_state_->pool), kVulkanObjectTypeDescriptorPool});
    // For the active slots, use set# to look up descriptorSet from boundDescriptorSets, and bind all of that descriptor set's
    // resources
    for (auto binding_req_pair : binding_req_map) {
        auto binding = binding_req_pair.first;
        // We aren't validating descriptors created with PARTIALLY_BOUND or UPDATE_AFTER_BIND, so don't record state
        if (p_layout_->GetDescriptorBindingFlagsFromBinding(binding) &
            (VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT)) {
            continue;
        }
        auto range = p_layout_->GetGlobalIndexRangeFromBinding(binding);
        for (uint32_t i = range.start; i < range.end; ++i) {
            descriptors_[i]->UpdateDrawState(device_data, cb_node);
        }
    }
}

void cvdescriptorset::DescriptorSet::FilterAndTrackOneBindingReq(const BindingReqMap::value_type &binding_req_pair,
                                                                 const BindingReqMap &in_req, BindingReqMap *out_req,
                                                                 TrackedBindings *bindings) {
    assert(out_req);
    assert(bindings);
    const auto binding = binding_req_pair.first;
    // Use insert and look at the boolean ("was inserted") in the returned pair to see if this is a new set member.
    // Saves one hash lookup vs. find ... compare w/ end ... insert.
    const auto it_bool_pair = bindings->insert(binding);
    if (it_bool_pair.second) {
        out_req->emplace(binding_req_pair);
    }
}

void cvdescriptorset::DescriptorSet::FilterAndTrackOneBindingReq(const BindingReqMap::value_type &binding_req_pair,
                                                                 const BindingReqMap &in_req, BindingReqMap *out_req,
                                                                 TrackedBindings *bindings, uint32_t limit) {
    if (bindings->size() < limit) FilterAndTrackOneBindingReq(binding_req_pair, in_req, out_req, bindings);
}

void cvdescriptorset::DescriptorSet::FilterAndTrackBindingReqs(CMD_BUFFER_STATE *cb_state, const BindingReqMap &in_req,
                                                               BindingReqMap *out_req) {
    TrackedBindings &bound = cached_validation_[cb_state].command_binding_and_usage;
    if (bound.size() == GetBindingCount()) {
        return;  // All bindings are bound, out req is empty
    }
    for (const auto &binding_req_pair : in_req) {
        const auto binding = binding_req_pair.first;
        // If a binding doesn't exist, or has already been bound, skip it
        if (p_layout_->HasBinding(binding)) {
            FilterAndTrackOneBindingReq(binding_req_pair, in_req, out_req, &bound);
        }
    }
}

void cvdescriptorset::DescriptorSet::FilterAndTrackBindingReqs(CMD_BUFFER_STATE *cb_state, PIPELINE_STATE *pipeline,
                                                               const BindingReqMap &in_req, BindingReqMap *out_req) {
    auto &validated = cached_validation_[cb_state];
    auto &image_sample_val = validated.image_samplers[pipeline];
    auto *const dynamic_buffers = &validated.dynamic_buffers;
    auto *const non_dynamic_buffers = &validated.non_dynamic_buffers;
    const auto &stats = p_layout_->GetBindingTypeStats();
    for (const auto &binding_req_pair : in_req) {
        auto binding = binding_req_pair.first;
        VkDescriptorSetLayoutBinding const *layout_binding = p_layout_->GetDescriptorSetLayoutBindingPtrFromBinding(binding);
        if (!layout_binding) {
            continue;
        }
        // Caching criteria differs per type.
        // If image_layout have changed , the image descriptors need to be validated against them.
        if ((layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) ||
            (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)) {
            FilterAndTrackOneBindingReq(binding_req_pair, in_req, out_req, dynamic_buffers, stats.dynamic_buffer_count);
        } else if ((layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) ||
                   (layout_binding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)) {
            FilterAndTrackOneBindingReq(binding_req_pair, in_req, out_req, non_dynamic_buffers, stats.non_dynamic_buffer_count);
        } else {
            // This is rather crude, as the changed layouts may not impact the bound descriptors,
            // but the simple "versioning" is a simple "dirt" test.
            auto &version = image_sample_val[binding];  // Take advantage of default construtor zero initialzing new entries
            if (version != cb_state->image_layout_change_count) {
                version = cb_state->image_layout_change_count;
                out_req->emplace(binding_req_pair);
            }
        }
    }
}









// This is a helper function that iterates over a set of Write and Copy updates, pulls the DescriptorSet* for updated
//  sets, and then calls their respective Perform[Write|Copy]Update functions.
// Prerequisite : ValidateUpdateDescriptorSets() should be called and return "false" prior to calling PerformUpdateDescriptorSets()
//  with the same set of updates.
// This is split from the validate code to allow validation prior to calling down the chain, and then update after
//  calling down the chain.

cvdescriptorset::DecodedTemplateUpdate::DecodedTemplateUpdate(CoreChecks *device_data, VkDescriptorSet descriptorSet,
                                                              const TEMPLATE_STATE *template_state, const void *pData,
                                                              VkDescriptorSetLayout push_layout) {
    auto const &create_info = template_state->create_info;
    inline_infos.resize(create_info.descriptorUpdateEntryCount);  // Make sure we have one if we need it
    desc_writes.reserve(create_info.descriptorUpdateEntryCount);  // emplaced, so reserved without initialization
    VkDescriptorSetLayout effective_dsl = create_info.templateType == VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET
                                              ? create_info.descriptorSetLayout
                                              : push_layout;
    auto layout_obj = GetDescriptorSetLayout(device_data, effective_dsl);

    // Create a WriteDescriptorSet struct for each template update entry
    for (uint32_t i = 0; i < create_info.descriptorUpdateEntryCount; i++) {
        auto binding_count = layout_obj->GetDescriptorCountFromBinding(create_info.pDescriptorUpdateEntries[i].dstBinding);
        auto binding_being_updated = create_info.pDescriptorUpdateEntries[i].dstBinding;
        auto dst_array_element = create_info.pDescriptorUpdateEntries[i].dstArrayElement;

        desc_writes.reserve(desc_writes.size() + create_info.pDescriptorUpdateEntries[i].descriptorCount);
        for (uint32_t j = 0; j < create_info.pDescriptorUpdateEntries[i].descriptorCount; j++) {
            desc_writes.emplace_back();
            auto &write_entry = desc_writes.back();

            size_t offset = create_info.pDescriptorUpdateEntries[i].offset + j * create_info.pDescriptorUpdateEntries[i].stride;
            char *update_entry = (char *)(pData) + offset;

            if (dst_array_element >= binding_count) {
                dst_array_element = 0;
                binding_being_updated = layout_obj->GetNextValidBinding(binding_being_updated);
            }

            write_entry.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_entry.pNext = NULL;
            write_entry.dstSet = descriptorSet;
            write_entry.dstBinding = binding_being_updated;
            write_entry.dstArrayElement = dst_array_element;
            write_entry.descriptorCount = 1;
            write_entry.descriptorType = create_info.pDescriptorUpdateEntries[i].descriptorType;

            switch (create_info.pDescriptorUpdateEntries[i].descriptorType) {
                case VK_DESCRIPTOR_TYPE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                    write_entry.pImageInfo = reinterpret_cast<VkDescriptorImageInfo *>(update_entry);
                    break;

                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    write_entry.pBufferInfo = reinterpret_cast<VkDescriptorBufferInfo *>(update_entry);
                    break;

                case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                    write_entry.pTexelBufferView = reinterpret_cast<VkBufferView *>(update_entry);
                    break;
                case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT: {
                    VkWriteDescriptorSetInlineUniformBlockEXT *inline_info = &inline_infos[i];
                    inline_info->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT;
                    inline_info->pNext = nullptr;
                    inline_info->dataSize = create_info.pDescriptorUpdateEntries[i].descriptorCount;
                    inline_info->pData = update_entry;
                    write_entry.pNext = inline_info;
                    // skip the rest of the array, they just represent bytes in the update
                    j = create_info.pDescriptorUpdateEntries[i].descriptorCount;
                    break;
                }
                default:
                    assert(0);
                    break;
            }
            dst_array_element++;
        }
    }
}

cvdescriptorset::BufferDescriptor::BufferDescriptor(const VkDescriptorType type)
    : storage_(false), dynamic_(false), buffer_(VK_NULL_HANDLE), offset_(0), range_(0) {
    updated = false;
    descriptor_class = GeneralBuffer;
    if (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC == type) {
        dynamic_ = true;
    } else if (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER == type) {
        storage_ = true;
    } else if (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC == type) {
        dynamic_ = true;
        storage_ = true;
    }
}
void cvdescriptorset::BufferDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    const auto &buffer_info = update->pBufferInfo[index];
    buffer_ = buffer_info.buffer;
    offset_ = buffer_info.offset;
    range_ = buffer_info.range;
}

void cvdescriptorset::BufferDescriptor::CopyUpdate(const Descriptor *src) {
    auto buff_desc = static_cast<const BufferDescriptor *>(src);
    updated = true;
    buffer_ = buff_desc->buffer_;
    offset_ = buff_desc->offset_;
    range_ = buff_desc->range_;
}

void cvdescriptorset::BufferDescriptor::UpdateDrawState(CoreChecks *dev_data, CMD_BUFFER_STATE *cb_node) {
}


cvdescriptorset::TexelDescriptor::TexelDescriptor(const VkDescriptorType type) : buffer_view_(VK_NULL_HANDLE), storage_(false) {
    updated = false;
    descriptor_class = TexelBuffer;
    if (VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER == type) storage_ = true;
}

void cvdescriptorset::TexelDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    buffer_view_ = update->pTexelBufferView[index];
}

void cvdescriptorset::TexelDescriptor::CopyUpdate(const Descriptor *src) {
    updated = true;
    buffer_view_ = static_cast<const TexelDescriptor *>(src)->buffer_view_;
}

void cvdescriptorset::TexelDescriptor::UpdateDrawState(CoreChecks *dev_data, CMD_BUFFER_STATE *cb_node) {
}


cvdescriptorset::ImageDescriptor::ImageDescriptor(const VkDescriptorType type)
    : storage_(false), image_view_(VK_NULL_HANDLE), image_layout_(VK_IMAGE_LAYOUT_UNDEFINED) {
    updated = false;
    descriptor_class = Image;
    if (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE == type) storage_ = true;
}

void cvdescriptorset::ImageDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    const auto &image_info = update->pImageInfo[index];
    image_view_ = image_info.imageView;
    image_layout_ = image_info.imageLayout;
}

void cvdescriptorset::ImageDescriptor::CopyUpdate(const Descriptor *src) {
    auto image_view = static_cast<const ImageDescriptor *>(src)->image_view_;
    auto image_layout = static_cast<const ImageDescriptor *>(src)->image_layout_;
    updated = true;
    image_view_ = image_view;
    image_layout_ = image_layout;
}

void cvdescriptorset::ImageDescriptor::UpdateDrawState(CoreChecks *dev_data, CMD_BUFFER_STATE *cb_node) {
}


void cvdescriptorset::SamplerDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    if (!immutable_) {
        sampler_ = update->pImageInfo[index].sampler;
    }
    updated = true;
}

void cvdescriptorset::SamplerDescriptor::CopyUpdate(const Descriptor *src) {
    if (!immutable_) {
        auto update_sampler = static_cast<const SamplerDescriptor *>(src)->sampler_;
        sampler_ = update_sampler;
    }
    updated = true;
}

void cvdescriptorset::SamplerDescriptor::UpdateDrawState(CoreChecks *dev_data, CMD_BUFFER_STATE *cb_node) {
}


cvdescriptorset::ImageSamplerDescriptor::ImageSamplerDescriptor(const VkSampler *immut)
    : sampler_(VK_NULL_HANDLE), immutable_(false), image_view_(VK_NULL_HANDLE), image_layout_(VK_IMAGE_LAYOUT_UNDEFINED) {
    updated = false;
    descriptor_class = ImageSampler;
    if (immut) {
        sampler_ = *immut;
        immutable_ = true;
    }
}

void cvdescriptorset::ImageSamplerDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    const auto &image_info = update->pImageInfo[index];
    if (!immutable_) {
        sampler_ = image_info.sampler;
    }
    image_view_ = image_info.imageView;
    image_layout_ = image_info.imageLayout;
}

void cvdescriptorset::ImageSamplerDescriptor::CopyUpdate(const Descriptor *src) {
    if (!immutable_) {
        auto update_sampler = static_cast<const ImageSamplerDescriptor *>(src)->sampler_;
        sampler_ = update_sampler;
    }
    auto image_view = static_cast<const ImageSamplerDescriptor *>(src)->image_view_;
    auto image_layout = static_cast<const ImageSamplerDescriptor *>(src)->image_layout_;
    updated = true;
    image_view_ = image_view;
    image_layout_ = image_layout;
}

void cvdescriptorset::ImageSamplerDescriptor::UpdateDrawState(CoreChecks *dev_data, CMD_BUFFER_STATE *cb_node) {
}

// This is a helper function that iterates over a set of Write and Copy updates, pulls the DescriptorSet* for updated
//  sets, and then calls their respective Perform[Write|Copy]Update functions.
// Prerequisite : ValidateUpdateDescriptorSets() should be called and return "false" prior to calling PerformUpdateDescriptorSets()
//  with the same set of updates.
// This is split from the validate code to allow validation prior to calling down the chain, and then update after
//  calling down the chain.
void cvdescriptorset::PerformUpdateDescriptorSets(CoreChecks *dev_data, uint32_t write_count, const VkWriteDescriptorSet *p_wds,
    uint32_t copy_count, const VkCopyDescriptorSet *p_cds) {
    // Write updates first
    uint32_t i = 0;
    for (i = 0; i < write_count; ++i) {
        auto dest_set = p_wds[i].dstSet;
        auto set_node = dev_data->GetSetNode(dest_set);
        if (set_node) {
            set_node->PerformWriteUpdate(&p_wds[i]);
        }
    }
    // Now copy updates
    for (i = 0; i < copy_count; ++i) {
        auto dst_set = p_cds[i].dstSet;
        auto src_set = p_cds[i].srcSet;
        auto src_node = dev_data->GetSetNode(src_set);
        auto dst_node = dev_data->GetSetNode(dst_set);
        if (src_node && dst_node) {
            dst_node->PerformCopyUpdate(&p_cds[i], src_node);
        }
    }
}


void CoreChecks::PerformUpdateDescriptorSetsWithTemplateKHR(VkDescriptorSet descriptorSet, const TEMPLATE_STATE *template_state,
                                                            const void *pData) {
    // Translate the templated update into a normal update for validation...
    cvdescriptorset::DecodedTemplateUpdate decoded_update(this, descriptorSet, template_state, pData);
    cvdescriptorset::PerformUpdateDescriptorSets(this, static_cast<uint32_t>(decoded_update.desc_writes.size()),
                                                 decoded_update.desc_writes.data(), 0, NULL);
}


// Update the common AllocateDescriptorSetsData
void CoreChecks::UpdateAllocateDescriptorSetsData(const VkDescriptorSetAllocateInfo *p_alloc_info,
                                                  cvdescriptorset::AllocateDescriptorSetsData *ds_data) {
    for (uint32_t i = 0; i < p_alloc_info->descriptorSetCount; i++) {
        auto layout = GetDescriptorSetLayout(this, p_alloc_info->pSetLayouts[i]);
        if (layout) {
            ds_data->layout_nodes[i] = layout;
            // Count total descriptors required per type
            for (uint32_t j = 0; j < layout->GetBindingCount(); ++j) {
                const auto &binding_layout = layout->GetDescriptorSetLayoutBindingPtrFromIndex(j);
                uint32_t typeIndex = static_cast<uint32_t>(binding_layout->descriptorType);
                ds_data->required_descriptors_by_type[typeIndex] += binding_layout->descriptorCount;
            }
        }
    }
}
// Decrement allocated sets from the pool and insert new sets into set_map
void CoreChecks::PerformAllocateDescriptorSets(const VkDescriptorSetAllocateInfo *p_alloc_info,
                                               const VkDescriptorSet *descriptor_sets,
                                               const cvdescriptorset::AllocateDescriptorSetsData *ds_data) {
    auto pool_state = descriptorPoolMap[p_alloc_info->descriptorPool].get();
    // Account for sets and individual descriptors allocated from pool
    pool_state->availableSets -= p_alloc_info->descriptorSetCount;
    for (auto it = ds_data->required_descriptors_by_type.begin(); it != ds_data->required_descriptors_by_type.end(); ++it) {
        pool_state->availableDescriptorTypeCount[it->first] -= ds_data->required_descriptors_by_type.at(it->first);
    }

    const auto *variable_count_info = lvl_find_in_chain<VkDescriptorSetVariableDescriptorCountAllocateInfoEXT>(p_alloc_info->pNext);
    bool variable_count_valid = variable_count_info && variable_count_info->descriptorSetCount == p_alloc_info->descriptorSetCount;

    // Create tracking object for each descriptor set; insert into global map and the pool's set.
    for (uint32_t i = 0; i < p_alloc_info->descriptorSetCount; i++) {
        uint32_t variable_count = variable_count_valid ? variable_count_info->pDescriptorCounts[i] : 0;

        std::unique_ptr<cvdescriptorset::DescriptorSet> new_ds(new cvdescriptorset::DescriptorSet(
            descriptor_sets[i], p_alloc_info->descriptorPool, ds_data->layout_nodes[i], variable_count, this));
        pool_state->sets.insert(new_ds.get());
        new_ds->in_use.store(0);
        setMap[descriptor_sets[i]] = std::move(new_ds);
    }
}

cvdescriptorset::PrefilterBindRequestMap::PrefilterBindRequestMap(cvdescriptorset::DescriptorSet &ds, const BindingReqMap &in_map,
                                                                  CMD_BUFFER_STATE *cb_state)
    : filtered_map_(), orig_map_(in_map) {
    if (ds.GetTotalDescriptorCount() > kManyDescriptors_) {
        filtered_map_.reset(new std::map<uint32_t, descriptor_req>());
        ds.FilterAndTrackBindingReqs(cb_state, orig_map_, filtered_map_.get());
    }
}
cvdescriptorset::PrefilterBindRequestMap::PrefilterBindRequestMap(cvdescriptorset::DescriptorSet &ds, const BindingReqMap &in_map,
                                                                  CMD_BUFFER_STATE *cb_state, PIPELINE_STATE *pipeline)
    : filtered_map_(), orig_map_(in_map) {
    if (ds.GetTotalDescriptorCount() > kManyDescriptors_) {
        filtered_map_.reset(new std::map<uint32_t, descriptor_req>());
        ds.FilterAndTrackBindingReqs(cb_state, pipeline, orig_map_, filtered_map_.get());
    }
}
